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
from collections import defaultdict

# from tqdm.notebook import tqdm
import time
import copy
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from numba import njit

#################################
# 修改sonata import路径
from ..models import sonata

#################################
sys.path.append("../P3-SAM")
from model import build_P3SAM, load_state_dict


class YSAM(nn.Module):
    def __init__(self):
        super().__init__()
        build_P3SAM(self)

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        assign=False,
        ignore_seg_mlp=False,
        ignore_seg_s2_mlp=False,
        ignore_iou_mlp=False,
    ):
        load_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
            ignore_seg_mlp=ignore_seg_mlp,
            ignore_seg_s2_mlp=ignore_seg_s2_mlp,
            ignore_iou_mlp=ignore_iou_mlp,
        )

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


# @njit
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


# @njit
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


def fix_label(face_ids, adjacent_faces, use_aabb=False, mesh=None, show_info=False):
    if use_aabb:

        def _cal_aabb(face_ids, i, _points_org):
            _part_mask = face_ids == i
            _faces = mesh.faces[_part_mask]
            _faces = np.reshape(_faces, (-1))
            _points = mesh.vertices[_faces]
            min_xyz, max_xyz = better_aabb(_points)
            _part_mask = (
                (_points_org[:, 0] >= min_xyz[0])
                & (_points_org[:, 0] <= max_xyz[0])
                & (_points_org[:, 1] >= min_xyz[1])
                & (_points_org[:, 1] <= max_xyz[1])
                & (_points_org[:, 2] >= min_xyz[2])
                & (_points_org[:, 2] <= max_xyz[2])
            )
            _part_mask = np.reshape(_part_mask, (-1, 3))
            _part_mask = np.all(_part_mask, axis=1)
            return i, [min_xyz, max_xyz], _part_mask

        with Timer("计算aabb"):
            aabb = {}
            unique_ids = np.unique(face_ids)
            # print(max(unique_ids))
            aabb_face_mask = {}
            _faces = mesh.faces
            _vertices = mesh.vertices
            _faces = np.reshape(_faces, (-1))
            _points = _vertices[_faces]
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = []
                for i in unique_ids:
                    if i < 0:
                        continue
                    futures.append(executor.submit(_cal_aabb, face_ids, i, _points))
                for future in futures:
                    res = future.result()
                    aabb[res[0]] = res[1]
                    aabb_face_mask[res[0]] = res[2]

            # _faces = mesh.faces
            # _vertices = mesh.vertices
            # _faces = np.reshape(_faces, (-1))
            # _points = _vertices[_faces]
            # aabb_face_mask = cal_aabb_mask(_points, face_ids)

    with Timer("合并mesh"):
        loop_cnt = 1
        changed = True
        progress = tqdm(disable=not show_info)
        no_mask_ids = np.where(face_ids < 0)[0].tolist()
        faces_max = adjacent_faces.shape[0]
        while changed and loop_cnt <= 50:
            changed = False
            # 获取无色面片
            new_no_mask_ids = []
            for i in no_mask_ids:
                # if face_ids[i] < 0:
                # 找邻居
                if not (0 <= i < faces_max):
                    continue
                _adj_faces = adjacent_faces[i]
                _adj_ids = []
                for j in _adj_faces:
                    if j == -1:
                        break
                    if face_ids[j] >= 0:
                        _tar_id = face_ids[j]
                        if use_aabb:
                            _mask = aabb_face_mask[_tar_id]
                            if _mask[i]:
                                _adj_ids.append(_tar_id)
                        else:
                            _adj_ids.append(_tar_id)
                if len(_adj_ids) == 0:
                    new_no_mask_ids.append(i)
                    continue
                _max_id = np.argmax(np.bincount(_adj_ids))
                face_ids[i] = _max_id
                changed = True
            no_mask_ids = new_no_mask_ids
            # print(loop_cnt)
            progress.update(1)
            # progress.set_description(f"合并mesh循环：{loop_cnt} {np.sum(face_ids < 0)}")
            loop_cnt += 1
    return face_ids


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


def calculate_face_areas(mesh):
    """
    计算每个三角形面片的面积
    :param mesh: trimesh.Trimesh 对象
    :return: 面片面积数组 (n_faces,)
    """
    return mesh.area_faces
    # # 提取顶点和面片索引
    # vertices = mesh.vertices
    # faces = mesh.faces

    # # 获取所有三个顶点的坐标
    # v0 = vertices[faces[:, 0]]
    # v1 = vertices[faces[:, 1]]
    # v2 = vertices[faces[:, 2]]

    # # 计算两个边向量
    # edge1 = v1 - v0
    # edge2 = v2 - v0

    # # 计算叉积的模长（向量面积的两倍）
    # cross_product = np.cross(edge1, edge2)
    # areas = 0.5 * np.linalg.norm(cross_product, axis=1)

    # return areas


def get_connected_region(face_ids, adjacent_faces, return_face_part_ids=False):
    vis = [False] * len(face_ids)
    parts = []
    face_part_ids = np.ones_like(face_ids) * -1
    for i in range(len(face_ids)):
        if vis[i]:
            continue
        _part = []
        _queue = [i]
        while len(_queue) > 0:
            _cur_face = _queue.pop(0)
            if vis[_cur_face]:
                continue
            vis[_cur_face] = True
            _part.append(_cur_face)
            face_part_ids[_cur_face] = len(parts)
            if not (0 <= _cur_face < adjacent_faces.shape[0]):
                continue
            _cur_face_id = face_ids[_cur_face]
            _adj_faces = adjacent_faces[_cur_face]
            for j in _adj_faces:
                if j == -1:
                    break
                if not vis[j] and face_ids[j] == _cur_face_id:
                    _queue.append(j)
        parts.append(_part)
    if return_face_part_ids:
        return parts, face_part_ids
    else:
        return parts


def aabb_distance(box1, box2):
    """
    计算两个轴对齐包围盒（AABB）之间的最近距离。
    :param box1: 元组 (min_x, min_y, min_z, max_x, max_y, max_z)
    :param box2: 元组 (min_x, min_y, min_z, max_x, max_y, max_z)
    :return: 最近距离（浮点数）
    """
    # 解包坐标
    min1, max1 = box1
    min2, max2 = box2

    # 计算各轴上的分离距离
    dx = max(0, max2[0] - min1[0], max1[0] - min2[0])  # x轴分离距离
    dy = max(0, max2[1] - min1[1], max1[1] - min2[1])  # y轴分离距离
    dz = max(0, max2[2] - min1[2], max1[2] - min2[2])  # z轴分离距离

    # 如果所有轴都重叠，则距离为0
    if dx == 0 and dy == 0 and dz == 0:
        return 0.0

    # 计算欧几里得距离
    return np.sqrt(dx**2 + dy**2 + dz**2)


def aabb_volume(aabb):
    """
    计算轴对齐包围盒（AABB）的体积。
    :param aabb: 元组 (min_x, min_y, min_z, max_x, max_y, max_z)
    :return: 体积（浮点数）
    """
    # 解包坐标
    min_xyz, max_xyz = aabb

    # 计算体积
    dx = max_xyz[0] - min_xyz[0]
    dy = max_xyz[1] - min_xyz[1]
    dz = max_xyz[2] - min_xyz[2]
    return dx * dy * dz


def find_neighbor_part(parts, adjacent_faces, parts_aabb=None, parts_ids=None):
    face2part = {}
    for i, part in enumerate(parts):
        for face in part:
            face2part[face] = i
    neighbor_parts = []
    for i, part in enumerate(parts):
        neighbor_part = set()
        for face in part:
            if not (0 <= face < adjacent_faces.shape[0]):
                continue
            for adj_face in adjacent_faces[face]:
                if adj_face == -1:
                    break
                if adj_face not in face2part:
                    continue
                if face2part[adj_face] == i:
                    continue
                if parts_ids is not None and parts_ids[face2part[adj_face]] in [-1, -2]:
                    continue
                neighbor_part.add(face2part[adj_face])
        neighbor_part = list(neighbor_part)
        if (
            parts_aabb is not None
            and parts_ids is not None
            and (parts_ids[i] == -1 or parts_ids[i] == -2)
            and len(neighbor_part) == 0
        ):
            min_dis = np.inf
            min_idx = -1
            for j, _part in enumerate(parts):
                if j == i:
                    continue
                if parts_ids[j] == -1 or parts_ids[j] == -2:
                    continue
                aabb_1 = parts_aabb[i]
                aabb_2 = parts_aabb[j]
                dis = aabb_distance(aabb_1, aabb_2)
                if dis < min_dis:
                    min_dis = dis
                    min_idx = j
                elif dis == min_dis:
                    if aabb_volume(parts_aabb[j]) < aabb_volume(parts_aabb[min_idx]):
                        min_idx = j
            neighbor_part = [min_idx]
        neighbor_parts.append(neighbor_part)
    return neighbor_parts


def do_post_process(
    face_areas, parts, adjacent_faces, face_ids, threshold=0.95, show_info=False
):
    # # 获取邻接面片
    # mesh_save = mesh.copy()
    # face_adjacency = mesh.face_adjacency
    # adjacent_faces = {}
    # for face1, face2 in face_adjacency:
    #     if face1 not in adjacent_faces:
    #         adjacent_faces[face1] = []
    #     if face2 not in adjacent_faces:
    #         adjacent_faces[face2] = []
    #     adjacent_faces[face1].append(face2)
    #     adjacent_faces[face2].append(face1)

    # parts = get_connected_region(face_ids, adjacent_faces)

    unique_ids = np.unique(face_ids)
    if show_info:
        print(f"连通区域数量：{len(parts)}")
        print(f"ID数量：{len(unique_ids)}")

    # face_areas = calculate_face_areas(mesh)
    total_area = np.sum(face_areas)
    if show_info:
        print(f"总面积：{total_area}")
    part_areas = []
    for i, part in enumerate(parts):
        part_area = np.sum(face_areas[part])
        part_areas.append(float(part_area / total_area))

    sorted_parts = sorted(zip(part_areas, parts), key=lambda x: x[0], reverse=True)
    parts = [x[1] for x in sorted_parts]
    part_areas = [x[0] for x in sorted_parts]
    integral_part_areas = np.cumsum(part_areas)

    neighbor_parts = find_neighbor_part(parts, adjacent_faces)

    new_face_ids = face_ids.copy()

    for i, part in enumerate(parts):
        if integral_part_areas[i] > threshold and part_areas[i] < 0.01:
            if len(neighbor_parts[i]) > 0:
                max_area = 0
                max_part = -1
                for j in neighbor_parts[i]:
                    if integral_part_areas[j] > threshold:
                        continue
                    if part_areas[j] > max_area:
                        max_area = part_areas[j]
                        max_part = j
                if max_part != -1:
                    if show_info:
                        print(f"合并mesh：{i} {max_part}")
                    parts[max_part].extend(part)
                    parts[i] = []
                    target_face_id = face_ids[parts[max_part][0]]
                    for face in part:
                        new_face_ids[face] = target_face_id

    return new_face_ids


def do_no_mask_process(parts, face_ids):
    # # 获取邻接面片
    # mesh_save = mesh.copy()
    # face_adjacency = mesh.face_adjacency
    # adjacent_faces = {}
    # for face1, face2 in face_adjacency:
    #     if face1 not in adjacent_faces:
    #         adjacent_faces[face1] = []
    #     if face2 not in adjacent_faces:
    #         adjacent_faces[face2] = []
    #     adjacent_faces[face1].append(face2)
    #     adjacent_faces[face2].append(face1)
    # parts = get_connected_region(face_ids, adjacent_faces)

    unique_ids = np.unique(face_ids)
    max_id = np.max(unique_ids)
    if -1 or -2 in unique_ids:
        new_face_ids = face_ids.copy()
        for i, part in enumerate(parts):
            if face_ids[part[0]] == -1 or face_ids[part[0]] == -2:
                for face in part:
                    new_face_ids[face] = max_id + 1
                max_id += 1
        return new_face_ids
    else:
        return face_ids


def union_aabb(aabb1, aabb2):
    min_xyz1 = aabb1[0]
    max_xyz1 = aabb1[1]
    min_xyz2 = aabb2[0]
    max_xyz2 = aabb2[1]
    min_xyz = np.minimum(min_xyz1, min_xyz2)
    max_xyz = np.maximum(max_xyz1, max_xyz2)
    return [min_xyz, max_xyz]


def aabb_increase(aabb1, aabb2):
    min_xyz_before = aabb1[0]
    max_xyz_before = aabb1[1]
    min_xyz_after, max_xyz_after = union_aabb(aabb1, aabb2)
    min_xyz_increase = np.abs(min_xyz_after - min_xyz_before) / np.abs(min_xyz_before)
    max_xyz_increase = np.abs(max_xyz_after - max_xyz_before) / np.abs(max_xyz_before)
    return min_xyz_increase, max_xyz_increase


def sort_multi_list(multi_list, key=lambda x: x[0], reverse=False):
    """
    multi_list: [list1, list2, list3, list4, ...], len(list1)=N, len(list2)=N, len(list3)=N, ...
    key: 排序函数，默认按第一个元素排序
    reverse: 排序顺序，默认降序
    return:
        [list1, list2, list3, list4, ...]: 按同一个顺序排序后的多个list
    """
    sorted_list = sorted(zip(*multi_list), key=key, reverse=reverse)
    return zip(*sorted_list)


# def sample_mesh(mesh, adjacent_faces, point_num=100000):
#     connected_parts = get_connected_region(np.ones(len(mesh.faces)), adjacent_faces)
#     _points, face_idx = trimesh.sample.sample_surface(mesh, point_num)
#     face_sampled = np.zeros(len(mesh.faces), dtype=np.bool)
#     face_sampled[face_idx] = True
#     for parts in connected_parts

# def parallel_run(model_parallel, feats, points, prompts):
#     bs = prompts.shape[0]
#     prompts_1 = prompts[:bs//2]
#     prompts_2 = prompts[bs//2:]
#     device_1 = 'cuda:0'
#     device_2 = 'cuda:1'
#     pred_mask_1_1, pred_mask_2_1, pred_mask_3_1, pred_iou_1 = get_mask(
#         model_parallel.module.to(device_1), feats, points, prompts_1, device=device_1
#     )
#     pred_mask_1_2, pred_mask_2_2, pred_mask_3_2, pred_iou_2 = get_mask(
#         model_parallel.module.to(device_2), feats, points, prompts_2, device=device_2
#     )
#     pred_mask_1 = np.concatenate([pred_mask_1_1, pred_mask_1_2], axis=1)
#     pred_mask_2 = np.concatenate([pred_mask_2_1, pred_mask_2_2], axis=1)
#     pred_mask_3 = np.concatenate([pred_mask_3_1, pred_mask_3_2], axis=1)
#     pred_iou = np.concatenate([pred_iou_1, pred_iou_2], axis=0)
#     return pred_mask_1, pred_mask_2, pred_mask_3, pred_iou

############################################################################################


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


###################### NUMBA 加速 ######################
@njit
def build_adjacent_faces_numba(face_adjacency):
    """
    使用 Numba 加速构建邻接面片数组。
    :param face_adjacency: (N, 2) numpy 数组，包含邻接面片对。
    :return:
        - adj_list: 一维数组，存储所有邻接面片。
        - offsets: 一维数组，记录每个面片的邻接起始位置。
    """
    n_faces = np.max(face_adjacency) + 1  # 总面片数
    n_edges = face_adjacency.shape[0]  # 总邻接边数

    # 第一步：统计每个面片的邻接数量（度数）
    degrees = np.zeros(n_faces, dtype=np.int32)
    for i in range(n_edges):
        f1, f2 = face_adjacency[i]
        degrees[f1] += 1
        degrees[f2] += 1
    max_degree = np.max(degrees)  # 最大度数

    adjacent_faces = np.ones((n_faces, max_degree), dtype=np.int32) * -1  # 邻接面片数组
    adjacent_faces_count = np.zeros(n_faces, dtype=np.int32)  # 邻接面片计数器
    for i in range(n_edges):
        f1, f2 = face_adjacency[i]
        adjacent_faces[f1, adjacent_faces_count[f1]] = f2
        adjacent_faces_count[f1] += 1
        adjacent_faces[f2, adjacent_faces_count[f2]] = f1
        adjacent_faces_count[f2] += 1
    return adjacent_faces


###################### NUMBA 加速 ######################


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
    with Timer("获取邻接面片"):
        # 获取邻接面片
        face_adjacency = mesh.face_adjacency
    with Timer("处理邻接面片"):
        # adjacent_faces = defaultdict(list)
        # for face1, face2 in face_adjacency:
        #     adjacent_faces[face1].append(face2)
        #     adjacent_faces[face2].append(face1)
        # adj_list, offsets = build_adjacent_faces_numba(face_adjacency)
        adjacent_faces = build_adjacent_faces_numba(face_adjacency)
    # with Timer("处理邻接面片2"):
    #     adjacent_faces = to_adj_dict(adj_list, offsets)

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
        bs = 64
        step_num = prompt_num // bs + 1
        mask_res = []
        iou_res = []
        for i in tqdm(range(step_num), disable=not show_info):
            cur_propmt = _point_prompts[bs * i : bs * (i + 1)]
            # pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = get_mask(
            #     model, _feats, _points, cur_propmt
            # )
            # pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = model_parallel(
            #     _feats, _points, cur_propmt
            # )
            # pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = parallel_run(
            #     model_parallel, _feats, _points, cur_propmt
            # )
            pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = get_mask(
                model_parallel, _feats, _points, cur_propmt
            )
            # print(pred_mask_1.shape, pred_mask_2.shape, pred_mask_3.shape, pred_iou.shape)
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

    with Timer("投影Mesh并统计label"):
        # 保存mesh结果
        face_seg_res = {}
        for i in final_mask_sorted:
            _part_mask = result_mask == i
            _face_idx = face_idx[_part_mask]
            for k in _face_idx:
                if k not in face_seg_res:
                    face_seg_res[k] = []
                face_seg_res[k].append(i)
        _part_mask = result_mask == -1
        _face_idx = face_idx[_part_mask]
        for k in _face_idx:
            if k not in face_seg_res:
                face_seg_res[k] = []
            face_seg_res[k].append(-1)

        face_ids = -np.ones(len(mesh.faces), dtype=np.int64) * 2
        for i in tqdm(face_seg_res, leave=False, disable=True):
            _seg_ids = np.array(face_seg_res[i])
            # 获取最多的seg_id
            _max_id = np.argmax(np.bincount(_seg_ids + 2)) - 2
            face_ids[i] = _max_id
        face_ids_org = face_ids.copy()
    if show_info:
        print("生成face_ids完成")

    # 获取邻接面片
    # face_adjacency = mesh.face_adjacency
    # adjacent_faces = {}
    # for face1, face2 in face_adjacency:
    #     if face1 not in adjacent_faces:
    #         adjacent_faces[face1] = []
    #     if face2 not in adjacent_faces:
    #         adjacent_faces[face2] = []
    #     adjacent_faces[face1].append(face2)
    #     adjacent_faces[face2].append(face1)

    with Timer("第一次修复face_ids"):
        face_ids += 1
        # face_ids = fix_label(face_ids, adjacent_faces, use_aabb=True, mesh=mesh, show_info=show_info)
        face_ids = fix_label(face_ids, adjacent_faces, mesh=mesh, show_info=show_info)
        face_ids -= 1
    if show_info:
        print("修复face_ids完成")

    color_map[-1] = np.array([255, 0, 0], dtype=np.uint8)

    if save_mid_res:
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh.glb"), mesh, face_ids, color_map
        )
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh_org.glb"),
            mesh,
            face_ids_org,
            color_map,
        )
        if show_info:
            print("保存mesh结果完成")

    with Timer("计算连通区域"):
        face_areas = calculate_face_areas(mesh)
        mesh_total_area = np.sum(face_areas)
        parts = get_connected_region(face_ids, adjacent_faces)
        connected_parts, _face_connected_parts_ids = get_connected_region(
            np.ones_like(face_ids), adjacent_faces, return_face_part_ids=True
        )
    if show_info:
        print(f"共{len(parts)}个mesh")
    with Timer("排序连通区域"):
        parts_cp_idx = []
        for x in parts:
            _face_idx = x[0]
            parts_cp_idx.append(_face_connected_parts_ids[_face_idx])
        parts_cp_idx = np.array(parts_cp_idx)
        parts_areas = [float(np.sum(face_areas[x])) for x in parts]
        connected_parts_areas = [float(np.sum(face_areas[x])) for x in connected_parts]
        parts_cp_areas = [connected_parts_areas[x] for x in parts_cp_idx]
        parts_sorted, parts_areas_sorted, parts_cp_areas_sorted = sort_multi_list(
            [parts, parts_areas, parts_cp_areas], key=lambda x: x[1], reverse=True
        )

    with Timer("去除面积过小的区域"):
        filtered_parts = []
        other_parts = []
        for i in range(len(parts_sorted)):
            parts = parts_sorted[i]
            area = parts_areas_sorted[i]
            cp_area = parts_cp_areas_sorted[i]
            if area / (cp_area + 1e-7) > 0.001:
                filtered_parts.append(i)
            else:
                other_parts.append(i)
    if show_info:
        print(f"保留{len(filtered_parts)}个mesh, 其他{len(other_parts)}个mesh")

    with Timer("去除面积过小区域的label"):
        face_ids_2 = face_ids.copy()
        part_num = len(cluster2.keys())
        for j in other_parts:
            parts = parts_sorted[j]
            for i in parts:
                face_ids_2[i] = -1

    with Timer("第二次修复face_ids"):
        face_ids_3 = face_ids_2.copy()
        # face_ids_3 = fix_label(face_ids_3, adjacent_faces, use_aabb=True, mesh=mesh, show_info=show_info)
        face_ids_3 = fix_label(
            face_ids_3, adjacent_faces, mesh=mesh, show_info=show_info
        )

    if save_mid_res:
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh_filtered_2.glb"),
            mesh,
            face_ids_3,
            color_map,
        )
        if show_info:
            print("保存mesh结果完成")

    with Timer("第二次计算连通区域"):
        parts_2 = get_connected_region(face_ids_3, adjacent_faces)
        parts_areas_2 = [float(np.sum(face_areas[x])) for x in parts_2]
        parts_ids_2 = [face_ids_3[x[0]] for x in parts_2]

    with Timer("添加过大的缺失part"):
        color_map_2 = copy.deepcopy(color_map)
        max_id = np.max(parts_ids_2)
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _area = parts_areas_2[i]
            _parts_id = face_ids_3[_parts[0]]
            if _area / mesh_total_area > 0.001:
                if _parts_id == -1 or _parts_id == -2:
                    parts_ids_2[i] = max_id + 1
                    max_id += 1
                    color_map_2[max_id] = np.random.rand(3) * 255
                    if show_info:
                        print(f"新增part {max_id}")
            # else:
            #     parts_ids_2[i] = -1

    with Timer("赋值新的face_ids"):
        face_ids_4 = face_ids_3.copy()
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _parts_id = parts_ids_2[i]
            for j in _parts:
                face_ids_4[j] = _parts_id
    with Timer("计算part和label的aabb"):
        ids_aabb = {}
        unique_ids = np.unique(face_ids_4)
        for i in unique_ids:
            if i < 0:
                continue
            _part_mask = face_ids_4 == i
            _faces = mesh.faces[_part_mask]
            _faces = np.reshape(_faces, (-1))
            _points = mesh.vertices[_faces]
            min_xyz = np.min(_points, axis=0)
            max_xyz = np.max(_points, axis=0)
            ids_aabb[i] = [min_xyz, max_xyz]

        parts_2_aabb = []
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _faces = mesh.faces[_parts]
            _faces = np.reshape(_faces, (-1))
            _points = mesh.vertices[_faces]
            min_xyz = np.min(_points, axis=0)
            max_xyz = np.max(_points, axis=0)
            parts_2_aabb.append([min_xyz, max_xyz])

    with Timer("计算part的邻居"):
        parts_2_neighbor = find_neighbor_part(
            parts_2, adjacent_faces, parts_2_aabb, parts_ids_2
        )
    with Timer("合并无mask区域"):
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _ids = parts_ids_2[i]
            if _ids == -1 or _ids == -2:
                _cur_aabb = parts_2_aabb[i]
                _min_aabb_increase = 1e10
                _min_id = -1
                for j in parts_2_neighbor[i]:
                    if parts_ids_2[j] == -1 or parts_ids_2[j] == -2:
                        continue
                    _tar_id = parts_ids_2[j]
                    _tar_aabb = ids_aabb[_tar_id]
                    _min_increase, _max_increase = aabb_increase(_tar_aabb, _cur_aabb)
                    _increase = max(np.max(_min_increase), np.max(_max_increase))
                    if _min_aabb_increase > _increase:
                        _min_aabb_increase = _increase
                        _min_id = _tar_id
                if _min_id >= 0:
                    parts_ids_2[i] = _min_id

    with Timer("再次赋值新的face_ids"):
        face_ids_4 = face_ids_3.copy()
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _parts_id = parts_ids_2[i]
            for j in _parts:
                face_ids_4[j] = _parts_id

    final_face_ids = face_ids_4
    if save_mid_res:
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh_final.glb"),
            mesh,
            face_ids_4,
            color_map_2,
        )

    if post_process:
        parts = get_connected_region(final_face_ids, adjacent_faces)
        final_face_ids = do_no_mask_process(parts, final_face_ids)
        face_ids_5 = do_post_process(
            face_areas,
            parts,
            adjacent_faces,
            face_ids_4,
            threshold,
            show_info=show_info,
        )
        if save_mid_res:
            save_mesh(
                os.path.join(save_path, "auto_mask_mesh_final_post.glb"),
                mesh,
                face_ids_5,
                color_map_2,
            )
        final_face_ids = face_ids_5
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
    ):
        """
        ckpt_path: str, 模型路径
        point_num: int, 采样点数量
        prompt_num: int, 提示数量
        threshold: float, 阈值
        post_process: bool, 是否后处理
        """
        self.model = YSAM()
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu")["state_dict"]
        )
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
            [self.model, self.model_parallel],
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
        )
