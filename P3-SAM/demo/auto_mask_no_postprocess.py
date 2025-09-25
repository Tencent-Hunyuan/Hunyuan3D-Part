import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import trimesh
from sklearn.decomposition import PCA
import fpsample
from tqdm import tqdm
import threading
import random

# from tqdm.notebook import tqdm
import time
import copy
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict

import numba
from numba import njit

sys.path.append("..")
from model import build_P3SAM, load_state_dict

from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist

cmd_loss = chamfer_3DDist()


class P3SAM(nn.Module):
    def __init__(self):
        super().__init__()
        build_P3SAM(self)

    def load_state_dict(self, 
                        ckpt_path=None, 
                        state_dict=None, 
                        strict=True, 
                        assign=False, 
                        ignore_seg_mlp=False, 
                        ignore_seg_s2_mlp=False, 
                        ignore_iou_mlp=False):
        load_state_dict(self, 
                        ckpt_path=ckpt_path, 
                        state_dict=state_dict, 
                        strict=strict, 
                        assign=assign, 
                        ignore_seg_mlp=ignore_seg_mlp, 
                        ignore_seg_s2_mlp=ignore_seg_s2_mlp, 
                        ignore_iou_mlp=ignore_iou_mlp)
        
    def forward(self, feats, points, point_prompt, iter=1):
        """
        feats: [K, N, 512]
        points: [K, N, 3]
        point_prompt: [K, N, 3]
        """
        # print(feats.shape, points.shape, point_prompt.shape)
        point_num = points.shape[1]
        feats = feats.transpose(0, 1)  # [N, K, 512]
        points = points.transpose(0, 1)  # [N, K, 3]
        point_prompt = point_prompt.transpose(0, 1)  # [N, K, 3]
        feats_seg = torch.cat([feats, points, point_prompt], dim=-1)  # [N, K, 512+3+3]

        # 预测mask stage-1
        pred_mask_1 = self.seg_mlp_1(feats_seg).squeeze(-1)  # [N, K]
        pred_mask_2 = self.seg_mlp_2(feats_seg).squeeze(-1)  # [N, K]
        pred_mask_3 = self.seg_mlp_3(feats_seg).squeeze(-1)  # [N, K]
        pred_mask = torch.stack(
            [pred_mask_1, pred_mask_2, pred_mask_3], dim=-1
        )  # [N, K, 3]

        for _ in range(iter):
            # 预测mask stage-2
            feats_seg_2 = torch.cat([feats_seg, pred_mask], dim=-1)  # [N, K, 512+3+3+3]
            feats_seg_global = self.seg_s2_mlp_g(feats_seg_2)  # [N, K, 512]
            feats_seg_global = torch.max(feats_seg_global, dim=0).values  # [K, 512]
            feats_seg_global = feats_seg_global.unsqueeze(0).repeat(
                point_num, 1, 1
            )  # [N, K, 512]
            feats_seg_3 = torch.cat(
                [feats_seg_global, feats_seg_2], dim=-1
            )  # [N, K, 512+3+3+3+512]
            pred_mask_s2_1 = self.seg_s2_mlp_1(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2_2 = self.seg_s2_mlp_2(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2_3 = self.seg_s2_mlp_3(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2 = torch.stack(
                [pred_mask_s2_1, pred_mask_s2_2, pred_mask_s2_3], dim=-1
            )  # [N,, K 3]
            pred_mask = pred_mask_s2

        mask_1 = torch.sigmoid(pred_mask_s2_1).to(dtype=torch.float32)  # [N, K]
        mask_2 = torch.sigmoid(pred_mask_s2_2).to(dtype=torch.float32)  # [N, K]
        mask_3 = torch.sigmoid(pred_mask_s2_3).to(dtype=torch.float32)  # [N, K]

        feats_iou = torch.cat(
            [feats_seg_global, feats_seg, pred_mask_s2], dim=-1
        )  # [N, K, 512+3+3+3+512]
        feats_iou = self.iou_mlp(feats_iou)  # [N, K, 512]
        feats_iou = torch.max(feats_iou, dim=0).values  # [K, 512]
        pred_iou = self.iou_mlp_out(feats_iou)  # [K, 3]
        pred_iou = torch.sigmoid(pred_iou).to(dtype=torch.float32)  # [K, 3]

        mask_1 = mask_1.transpose(0, 1)  # [K, N]
        mask_2 = mask_2.transpose(0, 1)  # [K, N]
        mask_3 = mask_3.transpose(0, 1)  # [K, N]

        return mask_1, mask_2, mask_3, pred_iou


def normalize_pc(pc):
    """
    pc: (N, 3)
    """
    max_, min_ = np.max(pc, axis=0), np.min(pc, axis=0)
    center = (max_ + min_) / 2
    scale = (max_ - min_) / 2
    scale = np.max(np.abs(scale))
    pc = (pc - center) / (scale + 1e-10)
    return pc


@torch.no_grad()
def get_feat(model, points, normals):
    data_dict = {
        "coord": points,
        "normal": normals,
        "color": np.ones_like(points),
        "batch": np.zeros(points.shape[0], dtype=np.int64),
    }
    data_dict = model.transform(data_dict)
    for k in data_dict:
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].cuda()
    point = model.sonata(data_dict)
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    feat = point.feat  # [M, 1232]
    feat = model.mlp(feat)  # [M, 512]
    feat = feat[point.inverse]  # [N, 512]
    feats = feat
    return feats


@torch.no_grad()
def get_mask(model, feats, points, point_prompt, iter=1):
    """
    feats: [N, 512]
    points: [N, 3]
    point_prompt: [K, 3]
    """
    point_num = points.shape[0]
    prompt_num = point_prompt.shape[0]
    feats = feats.unsqueeze(1)  # [N, 1, 512]
    feats = feats.repeat(1, prompt_num, 1).cuda()  # [N, K, 512]
    points = torch.from_numpy(points).float().cuda().unsqueeze(1)  # [N, 1, 3]
    points = points.repeat(1, prompt_num, 1)  # [N, K, 3]
    prompt_coord = (
        torch.from_numpy(point_prompt).float().cuda().unsqueeze(0)
    )  # [1, K, 3]
    prompt_coord = prompt_coord.repeat(point_num, 1, 1)  # [N, K, 3]

    feats = feats.transpose(0, 1)  # [K, N, 512]
    points = points.transpose(0, 1)  # [K, N, 3]
    prompt_coord = prompt_coord.transpose(0, 1)  # [K, N, 3]

    mask_1, mask_2, mask_3, pred_iou = model(feats, points, prompt_coord, iter)

    mask_1 = mask_1.transpose(0, 1)  # [N, K]
    mask_2 = mask_2.transpose(0, 1)  # [N, K]
    mask_3 = mask_3.transpose(0, 1)  # [N, K]

    mask_1 = mask_1.detach().cpu().numpy() > 0.5
    mask_2 = mask_2.detach().cpu().numpy() > 0.5
    mask_3 = mask_3.detach().cpu().numpy() > 0.5

    org_iou = pred_iou.detach().cpu().numpy()  # [K, 3]

    return mask_1, mask_2, mask_3, org_iou


def cal_iou(m1, m2):
    return np.sum(np.logical_and(m1, m2)) / np.sum(np.logical_or(m1, m2))


def cal_single_iou(m1, m2):
    return np.sum(np.logical_and(m1, m2)) / np.sum(m1)


def iou_3d(box1, box2, signle=None):
    """
    计算两个三维边界框的交并比 (IoU)

    参数:
        box1 (list): 第一个边界框的坐标 [x1_min, y1_min, z1_min, x1_max, y1_max, z1_max]
        box2 (list): 第二个边界框的坐标 [x2_min, y2_min, z2_min, x2_max, y2_max, z2_max]

    返回:
        float: 交并比 (IoU) 值
    """
    # 计算交集的坐标
    intersection_xmin = max(box1[0], box2[0])
    intersection_ymin = max(box1[1], box2[1])
    intersection_zmin = max(box1[2], box2[2])
    intersection_xmax = min(box1[3], box2[3])
    intersection_ymax = min(box1[4], box2[4])
    intersection_zmax = min(box1[5], box2[5])

    # 判断是否有交集
    if (
        intersection_xmin >= intersection_xmax
        or intersection_ymin >= intersection_ymax
        or intersection_zmin >= intersection_zmax
    ):
        return 0.0  # 无交集

    # 计算交集的体积
    intersection_volume = (
        (intersection_xmax - intersection_xmin)
        * (intersection_ymax - intersection_ymin)
        * (intersection_zmax - intersection_zmin)
    )

    # 计算两个盒子的体积
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

    if signle is None:
        # 计算并集的体积
        union_volume = box1_volume + box2_volume - intersection_volume
    elif signle == "1":
        union_volume = box1_volume
    elif signle == "2":
        union_volume = box2_volume
    else:
        raise ValueError("signle must be None or 1 or 2")

    # 计算 IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0.0
    return iou


def cal_point_bbox_iou(p1, p2, signle=None):
    min_p1 = np.min(p1, axis=0)
    max_p1 = np.max(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p2 = np.max(p2, axis=0)
    box1 = [min_p1[0], min_p1[1], min_p1[2], max_p1[0], max_p1[1], max_p1[2]]
    box2 = [min_p2[0], min_p2[1], min_p2[2], max_p2[0], max_p2[1], max_p2[2]]
    return iou_3d(box1, box2, signle)


def cal_bbox_iou(points, m1, m2):
    p1 = points[m1]
    p2 = points[m2]
    return cal_point_bbox_iou(p1, p2)


def clean_mesh(mesh):
    """
    mesh: trimesh.Trimesh
    """
    # 1. 合并接近的顶点
    mesh.merge_vertices()

    # 2. 删除重复的顶点
    # 3. 删除重复的面片
    mesh.process(True)
    return mesh


def get_aabb_from_face_ids(mesh, face_ids):
    unique_ids = np.unique(face_ids)
    aabb = []
    for i in unique_ids:
        if i == -1 or i == -2:
            continue
        _part_mask = face_ids == i
        _faces = mesh.faces[_part_mask]
        _faces = np.reshape(_faces, (-1))
        _points = mesh.vertices[_faces]
        min_xyz = np.min(_points, axis=0)
        max_xyz = np.max(_points, axis=0)
        aabb.append([min_xyz, max_xyz])
    return np.array(aabb)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self  # 可以返回 self 以便在 with 块内访问

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f">>>>>>代码{self.name} 运行时间: {self.elapsed_time:.4f} 秒")


def sample_points_pre_face(vertices, faces, n_point_per_face=2000):
    n_f = faces.shape[0]  # 面片数量

    # 生成随机数 u, v
    u = np.sqrt(np.random.rand(n_f, n_point_per_face, 1))  # (n_f, n_point_per_face, 1)
    v = np.random.rand(n_f, n_point_per_face, 1)  # (n_f, n_point_per_face, 1)

    # 计算 barycentric 坐标
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v  # (n_f, n_point_per_face, 1)

    # 从顶点中提取每个面的三个顶点
    face_v_0 = vertices[faces[:, 0].reshape(-1)]  # (n_f, 3)
    face_v_1 = vertices[faces[:, 1].reshape(-1)]  # (n_f, 3)
    face_v_2 = vertices[faces[:, 2].reshape(-1)]  # (n_f, 3)

    # 扩展维度以匹配 w0, w1, w2 的形状
    face_v_0 = face_v_0.reshape(n_f, 1, 3)  # (n_f, 1, 3)
    face_v_1 = face_v_1.reshape(n_f, 1, 3)  # (n_f, 1, 3)
    face_v_2 = face_v_2.reshape(n_f, 1, 3)  # (n_f, 1, 3)

    # 计算每个点的坐标
    points = w0 * face_v_0 + w1 * face_v_1 + w2 * face_v_2  # (n_f, n_point_per_face, 3)

    return points


def cal_cd_batch(p1, p2, pn=100000):
    p1_n = p1.shape[0]
    batch_num = (p1_n + pn - 1) // pn
    p2_cuda = torch.from_numpy(p2).cuda().float().unsqueeze(0)
    p1_cuda = torch.from_numpy(p1).cuda().float().unsqueeze(0)
    cd_res = []
    for i in tqdm(range(batch_num)):
        start_idx = i * pn
        end_idx = min((i + 1) * pn, p1_n)
        _p1_cuda = p1_cuda[:, start_idx:end_idx, :]
        _, _, idx, _ = cmd_loss(_p1_cuda, p2_cuda)
        idx = idx[0].detach().cpu().numpy()
        cd_res.append(idx)
    cd_res = np.concatenate(cd_res, axis=0)
    return cd_res


def remove_outliers_iqr(data, factor=1.5):
    """
    基于 IQR 去除离群值
    :param data: 输入的列表或 NumPy 数组
    :param factor: IQR 的倍数（默认 1.5）
    :return: 去除离群值后的列表
    """
    data = np.array(data, dtype=np.float32)
    q1 = np.percentile(data, 25)  # 第一四分位数
    q3 = np.percentile(data, 75)  # 第三四分位数
    iqr = q3 - q1  # 四分位距
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)].tolist()


def better_aabb(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x = remove_outliers_iqr(x)
    y = remove_outliers_iqr(y)
    z = remove_outliers_iqr(z)
    min_xyz = np.array([np.min(x), np.min(y), np.min(z)])
    max_xyz = np.array([np.max(x), np.max(y), np.max(z)])
    return [min_xyz, max_xyz]


def save_mesh(save_path, mesh, face_ids, color_map):
    face_colors = np.zeros((len(mesh.faces), 3), dtype=np.uint8)
    for i in tqdm(range(len(mesh.faces)), disable=True):
        _max_id = face_ids[i]
        if _max_id == -2:
            continue
        face_colors[i, :3] = color_map[_max_id]

    mesh_save = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    mesh_save.visual.face_colors = face_colors
    mesh_save.export(save_path)
    mesh_save.export(save_path.replace(".glb", ".ply"))
    # print('保存mesh完成')

    scene_mesh = trimesh.Scene()
    scene_mesh.add_geometry(mesh_save)
    unique_ids = np.unique(face_ids)
    aabb = []
    for i in unique_ids:
        if i == -1 or i == -2:
            continue
        _part_mask = face_ids == i
        _faces = mesh.faces[_part_mask]
        _faces = np.reshape(_faces, (-1))
        _points = mesh.vertices[_faces]
        min_xyz, max_xyz = better_aabb(_points)
        center = (min_xyz + max_xyz) / 2
        size = max_xyz - min_xyz
        box = trimesh.path.creation.box_outline()
        box.vertices *= size
        box.vertices += center
        box_color = np.array([[color_map[i][0], color_map[i][1], color_map[i][2], 255]])
        box_color = np.repeat(box_color, len(box.entities), axis=0).astype(np.uint8)
        box.colors = box_color
        scene_mesh.add_geometry(box)
        min_xyz = np.min(_points, axis=0)
        max_xyz = np.max(_points, axis=0)
        aabb.append([min_xyz, max_xyz])
    scene_mesh.export(save_path.replace(".glb", "_aabb.glb"))
    aabb = np.array(aabb)
    np.save(save_path.replace(".glb", "_aabb.npy"), aabb)
    np.save(save_path.replace(".glb", "_face_ids.npy"), face_ids)


def mesh_sam(
    model,
    mesh,
    save_path,
    point_num=100000,
    prompt_num=400,
    save_mid_res=False,
    show_info=False,
    post_process=False,
    threshold=0.95,
    clean_mesh_flag=True,
    seed=42,
    prompt_bs=32,
):
    with Timer("加载mesh"):
        model, model_parallel = model
        if clean_mesh_flag:
            mesh = clean_mesh(mesh)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    if show_info:
        print(f"点数：{mesh.vertices.shape[0]} 面片数：{mesh.faces.shape[0]}")

    point_num = 100000
    prompt_num = 400

    with Timer("采样点云"):
        _points, face_idx = trimesh.sample.sample_surface(mesh, point_num, seed=seed)
        _points_org = _points.copy()
        _points = normalize_pc(_points)
        normals = mesh.face_normals[face_idx]
        # _points = _points + np.random.normal(0, 1, size=_points.shape) * 0.01
        # normals = normals * 0. # debug no normal
    if show_info:
        print(f"点数：{point_num} 面片数：{mesh.faces.shape[0]}")

    with Timer("获取特征"):
        _feats = get_feat(model, _points, normals)
    if show_info:
        print("预处理特征")

    if save_mid_res:
        feat_save = _feats.float().detach().cpu().numpy()
        data_scaled = feat_save / np.linalg.norm(feat_save, axis=-1, keepdims=True)
        pca = PCA(n_components=3)
        data_reduced = pca.fit_transform(data_scaled)
        data_reduced = (data_reduced - data_reduced.min()) / (
            data_reduced.max() - data_reduced.min()
        )
        _colors_pca = (data_reduced * 255).astype(np.uint8)
        pc_save = trimesh.points.PointCloud(_points, colors=_colors_pca)
        pc_save.export(os.path.join(save_path, "point_pca.glb"))
        pc_save.export(os.path.join(save_path, "point_pca.ply"))
        if show_info:
            print("PCA获取特征颜色")

    with Timer("FPS采样提示点"):
        fps_idx = fpsample.fps_sampling(_points, prompt_num)
        _point_prompts = _points[fps_idx]
    if save_mid_res:
        trimesh.points.PointCloud(_point_prompts, colors=_colors_pca[fps_idx]).export(
            os.path.join(save_path, "point_prompts_pca.glb")
        )
        trimesh.points.PointCloud(_point_prompts, colors=_colors_pca[fps_idx]).export(
            os.path.join(save_path, "point_prompts_pca.ply")
        )
    if show_info:
        print("采样完成")

    with Timer("推理"):
        bs = prompt_bs
        step_num = prompt_num // bs + 1
        mask_res = []
        iou_res = []
        for i in tqdm(range(step_num), disable=not show_info):
            cur_propmt = _point_prompts[bs * i : bs * (i + 1)]
            pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = get_mask(
                model_parallel, _feats, _points, cur_propmt
            )
            pred_mask = np.stack(
                [pred_mask_1, pred_mask_2, pred_mask_3], axis=-1
            )  # [N, K, 3]
            max_idx = np.argmax(pred_iou, axis=-1)  # [K]
            for j in range(max_idx.shape[0]):
                mask_res.append(pred_mask[:, j, max_idx[j]])
                iou_res.append(pred_iou[j, max_idx[j]])
    mask_res = np.stack(mask_res, axis=-1)  # [N, K]
    if show_info:
        print("prmopt 推理完成")

    with Timer("根据IOU排序"):
        iou_res = np.array(iou_res).tolist()
        mask_iou = [[mask_res[:, i], iou_res[i]] for i in range(prompt_num)]
        mask_iou_sorted = sorted(mask_iou, key=lambda x: x[1], reverse=True)
        mask_sorted = [mask_iou_sorted[i][0] for i in range(prompt_num)]
        iou_sorted = [mask_iou_sorted[i][1] for i in range(prompt_num)]

    # clusters = {}
    # for i in tqdm(range(prompt_num), desc="NMS", disable=not show_info):
    #     _mask = mask_sorted[i]
    #     union_flag = False
    #     for j in clusters.keys():
    #         if cal_iou(_mask, mask_sorted[j]) > 0.9:
    #             clusters[j].append(i)
    #             union_flag = True
    #             break
    #     if not union_flag:
    #         clusters[i] = [i]
    with Timer("NMS"):
        clusters = defaultdict(list)
        with ThreadPoolExecutor(max_workers=20) as executor:
            for i in tqdm(range(prompt_num), desc="NMS", disable=not show_info):
                _mask = mask_sorted[i]
                futures = []
                for j in clusters.keys():
                    futures.append(executor.submit(cal_iou, _mask, mask_sorted[j]))

                for j, future in zip(clusters.keys(), futures):
                    if future.result() > 0.9:
                        clusters[j].append(i)
                        break
                else:
                    clusters[i].append(i)

    # print(clusters)
    if show_info:
        print(f"NMS完成，mask数量：{len(clusters)}")

    if save_mid_res:
        part_mask_save_path = os.path.join(save_path, "part_mask")
        if os.path.exists(part_mask_save_path):
            shutil.rmtree(part_mask_save_path)
        os.makedirs(part_mask_save_path, exist_ok=True)
        for i in tqdm(clusters.keys(), desc="保存mask", disable=not show_info):
            cluster_num = len(clusters[i])
            cluster_iou = iou_sorted[i]
            cluster_area = np.sum(mask_sorted[i])
            if cluster_num <= 2:
                continue
            mask_save = mask_sorted[i]
            mask_save = np.expand_dims(mask_save, axis=-1)
            mask_save = np.repeat(mask_save, 3, axis=-1)
            mask_save = (mask_save * 255).astype(np.uint8)
            point_save = trimesh.points.PointCloud(_points, colors=mask_save)
            point_save.export(
                os.path.join(
                    part_mask_save_path,
                    f"mask_{i}_iou_{cluster_iou:.5f}_area_{cluster_area:.5f}_num_{cluster_num}.glb",
                )
            )

    # 过滤只有一个mask的cluster
    with Timer("过滤只有一个mask的cluster"):
        filtered_clusters = []
        other_clusters = []
        for i in clusters.keys():
            if len(clusters[i]) > 2:
                filtered_clusters.append(i)
            else:
                other_clusters.append(i)
    if show_info:
        print(
            f"过滤前：{len(clusters)} 个cluster，"
            f"过滤后：{len(filtered_clusters)} 个cluster"
        )

    # 再次合并
    with Timer("再次合并"):
        filtered_clusters_num = len(filtered_clusters)
        cluster2 = {}
        is_union = [False] * filtered_clusters_num
        for i in range(filtered_clusters_num):
            if is_union[i]:
                continue
            cur_cluster = filtered_clusters[i]
            cluster2[cur_cluster] = [cur_cluster]
            for j in range(i + 1, filtered_clusters_num):
                if is_union[j]:
                    continue
                tar_cluster = filtered_clusters[j]
                # if cal_single_iou(mask_sorted[tar_cluster], mask_sorted[cur_cluster]) > 0.9:
                # if cal_iou(mask_sorted[tar_cluster], mask_sorted[cur_cluster]) > 0.5:
                if (
                    cal_bbox_iou(
                        _points, mask_sorted[tar_cluster], mask_sorted[cur_cluster]
                    )
                    > 0.5
                ):
                    cluster2[cur_cluster].append(tar_cluster)
                    is_union[j] = True
    if show_info:
        print(f"再次合并，合并数量：{len(cluster2.keys())}")

    with Timer("计算没有mask的点"):
        no_mask = np.ones(point_num)
        for i in cluster2:
            part_mask = mask_sorted[i]
            no_mask[part_mask] = 0
    if show_info:
        print(
            f"{np.sum(no_mask == 1)} 个点没有mask,"
            f" 占比：{np.sum(no_mask == 1) / point_num:.4f}"
        )

    with Timer("修补遗漏mask"):
        # 查询漏掉的mask
        for i in tqdm(range(len(mask_sorted)), desc="漏掉mask", disable=not show_info):
            if i in cluster2:
                continue
            part_mask = mask_sorted[i]
            _iou = cal_single_iou(part_mask, no_mask)
            if _iou > 0.7:
                cluster2[i] = [i]
                no_mask[part_mask] = 0
                if save_mid_res:
                    mask_save = mask_sorted[i]
                    mask_save = np.expand_dims(mask_save, axis=-1)
                    mask_save = np.repeat(mask_save, 3, axis=-1)
                    mask_save = (mask_save * 255).astype(np.uint8)
                    point_save = trimesh.points.PointCloud(_points, colors=mask_save)
                    cluster_iou = iou_sorted[i]
                    cluster_area = int(np.sum(mask_sorted[i]))
                    cluster_num = 1
                    point_save.export(
                        os.path.join(
                            part_mask_save_path,
                            f"mask_{i}_iou_{cluster_iou:.5f}_area_{cluster_area:.5f}_num_{cluster_num}.glb",
                        )
                    )
    # print(cluster2)
    # print(len(cluster2.keys()))
    if show_info:
        print(f"修补遗漏mask：{len(cluster2.keys())}")

    with Timer("计算点云最终mask"):
        final_mask = list(cluster2.keys())
        final_mask_area = [int(np.sum(mask_sorted[i])) for i in final_mask]
        final_mask_area = [
            [final_mask[i], final_mask_area[i]] for i in range(len(final_mask))
        ]
        final_mask_area_sorted = sorted(
            final_mask_area, key=lambda x: x[1], reverse=True
        )
        final_mask_sorted = [
            final_mask_area_sorted[i][0] for i in range(len(final_mask_area))
        ]
        final_mask_area_sorted = [
            final_mask_area_sorted[i][1] for i in range(len(final_mask_area))
        ]
    # print(final_mask_sorted)
    # print(final_mask_area_sorted)
    if show_info:
        print(f"最终mask数量：{len(final_mask_sorted)}")

    with Timer("点云上色"):
        # 生成color map
        color_map = {}
        for i in final_mask_sorted:
            part_color = np.random.rand(3) * 255
            color_map[i] = part_color
        # print(color_map)

        result_mask = -np.ones(point_num, dtype=np.int64)
        for i in final_mask_sorted:
            part_mask = mask_sorted[i]
            result_mask[part_mask] = i
    if save_mid_res:
        # 保存点云结果
        result_colors = np.zeros_like(_colors_pca)
        for i in final_mask_sorted:
            part_color = color_map[i]
            part_mask = mask_sorted[i]
            result_colors[part_mask, :3] = part_color
        trimesh.points.PointCloud(_points, colors=result_colors).export(
            os.path.join(save_path, "auto_mask_cluster.glb")
        )
        trimesh.points.PointCloud(_points, colors=result_colors).export(
            os.path.join(save_path, "auto_mask_cluster.ply")
        )
        if show_info:
            print("保存点云完成")

    with Timer("后处理"):
        valid_mask = result_mask >= 0
        _org = _points_org[valid_mask]
        _results = result_mask[valid_mask]
        pre_face = 10
        _face_points = sample_points_pre_face(
            mesh.vertices, mesh.faces, n_point_per_face=pre_face
        )
        _face_points = np.reshape(_face_points, (len(mesh.faces) * pre_face, 3))
        _idx = cal_cd_batch(_face_points, _org)
        _idx_res = _results[_idx]
        _idx_res = np.reshape(_idx_res, (-1, pre_face))

        face_ids = []
        for i in range(len(mesh.faces)):
            _label = np.argmax(np.bincount(_idx_res[i] + 2)) - 2
            face_ids.append(_label)
        final_face_ids = np.array(face_ids)

    if save_mid_res:
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh_final.glb"),
            mesh,
            final_face_ids,
            color_map,
        )

    with Timer("计算最后的aabb"):
        aabb = get_aabb_from_face_ids(mesh, final_face_ids)
    return aabb, final_face_ids, mesh


class AutoMask:
    def __init__(
        self,
        ckpt_path,
        point_num=100000,
        prompt_num=400,
        threshold=0.95,
        post_process=True,
        automask_instance=None,
    ):
        """
        ckpt_path: str, 模型路径
        point_num: int, 采样点数量
        prompt_num: int, 提示数量
        threshold: float, 阈值
        post_process: bool, 是否后处理
        """
        if automask_instance is not None:
            self.model = automask_instance.model
            self.model_parallel = automask_instance.model_parallel
        else:
            self.model = P3SAM()
            self.model.load_state_dict(ckpt_path)
            self.model.eval()
            self.model_parallel = torch.nn.DataParallel(self.model)
            self.model.cuda()
            self.model_parallel.cuda()
        self.point_num = point_num
        self.prompt_num = prompt_num
        self.threshold = threshold
        self.post_process = post_process

    def predict_aabb(
        self,
        mesh,
        point_num=None,
        prompt_num=None,
        threshold=None,
        post_process=None,
        save_path=None,
        save_mid_res=False,
        show_info=True,
        clean_mesh_flag=True,
        seed=42,
        is_parallel=True,
        prompt_bs=32,
    ):
        """
        Parameters:
            mesh: trimesh.Trimesh, 输入网格
            point_num: int, 采样点数量
            prompt_num: int, 提示数量
            threshold: float, 阈值
            post_process: bool, 是否后处理
        Returns:
            aabb: np.ndarray, 包围盒
            face_ids: np.ndarray, 面id
        """
        point_num = point_num if point_num is not None else self.point_num
        prompt_num = prompt_num if prompt_num is not None else self.prompt_num
        threshold = threshold if threshold is not None else self.threshold
        post_process = post_process if post_process is not None else self.post_process
        return mesh_sam(
            [self.model, self.model_parallel if is_parallel else self.model],
            mesh,
            save_path=save_path,
            point_num=point_num,
            prompt_num=prompt_num,
            threshold=threshold,
            post_process=post_process,
            show_info=show_info,
            save_mid_res=save_mid_res,
            clean_mesh_flag=clean_mesh_flag,
            seed=seed,
            prompt_bs=prompt_bs,
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ckpt_path", type=str, default=None, help="模型路径"
    )
    argparser.add_argument(
        "--mesh_path", type=str, default="assets/1.glb", help="输入网格路径"
    )
    argparser.add_argument(
        "--output_path", type=str, default="results/1", help="保存路径"
    )
    argparser.add_argument("--point_num", type=int, default=100000, help="采样点数量")
    argparser.add_argument("--prompt_num", type=int, default=400, help="提示数量")
    argparser.add_argument("--threshold", type=float, default=0.95, help="阈值")
    argparser.add_argument("--post_process", type=int, default=0, help="是否后处理")
    argparser.add_argument(
        "--save_mid_res", type=int, default=1, help="是否保存中间结果"
    )
    argparser.add_argument("--show_info", type=int, default=1, help="是否显示信息")
    argparser.add_argument(
        "--show_time_info", type=int, default=1, help="是否显示时间信息"
    )
    argparser.add_argument("--seed", type=int, default=42, help="随机种子")
    argparser.add_argument("--parallel", type=int, default=1, help="是否使用多卡")
    argparser.add_argument(
        "--prompt_bs", type=int, default=32, help="提示点推理时的batch size大小"
    )
    argparser.add_argument("--clean_mesh", type=int, default=1, help="是否清洗网格")
    args = argparser.parse_args()
    Timer.STATE = args.show_time_info

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    ckpt_path = args.ckpt_path
    auto_mask = AutoMask(ckpt_path)
    mesh_path = args.mesh_path
    if os.path.isdir(mesh_path):
        for file in os.listdir(mesh_path):
            if not (
                file.endswith(".glb") or file.endswith(".obj") or file.endswith(".ply")
            ):
                continue
            _mesh_path = os.path.join(mesh_path, file)
            _output_path = os.path.join(output_path, file[:-4])
            os.makedirs(_output_path, exist_ok=True)
            mesh = trimesh.load(_mesh_path, force="mesh")
            set_seed(args.seed)
            aabb, face_ids, mesh = auto_mask.predict_aabb(
                mesh,
                save_path=_output_path,
                point_num=args.point_num,
                prompt_num=args.prompt_num,
                threshold=args.threshold,
                post_process=args.post_process,
                save_mid_res=args.save_mid_res,
                show_info=args.show_info,
                seed=args.seed,
                is_parallel=args.parallel,
                clean_mesh_flag=args.clean_mesh,
            )
    else:
        mesh = trimesh.load(mesh_path, force="mesh")
        set_seed(args.seed)
        aabb, face_ids, mesh = auto_mask.predict_aabb(
            mesh,
            save_path=output_path,
            point_num=args.point_num,
            prompt_num=args.prompt_num,
            threshold=args.threshold,
            post_process=args.post_process,
            save_mid_res=args.save_mid_res,
            show_info=args.show_info,
            seed=args.seed,
            is_parallel=args.parallel,
            clean_mesh_flag=args.clean_mesh,
        )

    ###############################################
    ## 可以通过以下代码保存返回的结果
    ## You can save the returned result by the following code
    ################# save result #################
    # color_map = {}
    # unique_ids = np.unique(face_ids)
    # for i in unique_ids:
    #     if i == -1:
    #         continue
    #     part_color = np.random.rand(3) * 255
    #     color_map[i] = part_color
    # face_colors = []
    # for i in face_ids:
    #     if i == -1:
    #         face_colors.append([0, 0, 0])
    #     else:
    #         face_colors.append(color_map[i])
    # face_colors = np.array(face_colors).astype(np.uint8)
    # mesh_save = mesh.copy()
    # mesh_save.visual.face_colors = face_colors
    # mesh_save.export(os.path.join(output_path, 'auto_mask_mesh.glb'))
    # scene_mesh = trimesh.Scene()
    # scene_mesh.add_geometry(mesh_save)
    # for i in range(len(aabb)):
    #     min_xyz, max_xyz = aabb[i]
    #     center = (min_xyz + max_xyz) / 2
    #     size = max_xyz - min_xyz
    #     box = trimesh.path.creation.box_outline()
    #     box.vertices *= size
    #     box.vertices += center
    #     scene_mesh.add_geometry(box)
    # scene_mesh.export(os.path.join(output_path, 'auto_mask_aabb.glb'))
    ################# save result #################

"""
python auto_mask_no_postprocess.py --parallel 0 
python auto_mask_no_postprocess.py --ckpt_path ../weights/p3sam.ckpt --mesh_path assets/1.glb --output_path results/1 --parallel 0 
python auto_mask_no_postprocess.py --ckpt_path ../weights/p3sam.ckpt --mesh_path assets --output_path results/all_no_postprocess 
"""
