import os 
import sys 
import torch
import torch.nn as nn
import numpy as np
import argparse
import viser
import trimesh
from sklearn.decomposition import PCA
import time

sys.path.append('..')
from model import build_P3SAM, load_state_dict

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

POINT_COLOR = np.array([255, 153, 153])
POINT_SIZE = 0.001
PROMPT_COLOR = np.array([0, 255, 0])
MASK_COLOR = np.array([0, 0, 255])

def normalize_pc(pc):
    '''
    pc: (N, 3)
    '''
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
        "batch": np.zeros(points.shape[0], dtype=np.int64)
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
    feat = point.feat # [M, 1232]
    feat = model.mlp(feat) # [M, 512]
    feat = feat[point.inverse] # [N, 512]
    feats = feat
    return feats

@torch.no_grad()
def get_mask(model, feats, points, point_prompt):
    point_num = points.shape[0]
    points = torch.from_numpy(points).float().cuda()   # [N, 3]
    prompt_coord = torch.from_numpy(point_prompt).float().cuda().unsqueeze(0)  # [1, 3]
    prompt_coord = prompt_coord.repeat(point_num, 1) # [N, 3]
    feats_seg = torch.cat([feats, points, prompt_coord], dim=-1) # [N, 512+3+3]

    # 预测mask stage-1
    pred_mask_1 = model.seg_mlp_1(feats_seg).squeeze(-1) # [N]
    pred_mask_2 = model.seg_mlp_2(feats_seg).squeeze(-1) # [N]
    pred_mask_3 = model.seg_mlp_3(feats_seg).squeeze(-1) # [N]
    pred_mask = torch.stack([pred_mask_1, pred_mask_2, pred_mask_3], dim=-1) # [N, 3]

    # 预测mask stage-2
    feats_seg_2 = torch.cat([feats_seg, pred_mask], dim=-1) # [N, 512+3+3+3]
    feats_seg_global = model.seg_s2_mlp_g(feats_seg_2) # [N, 512]
    feats_seg_global = torch.max(feats_seg_global, dim=0).values # [512]
    feats_seg_global = feats_seg_global.unsqueeze(0).repeat(point_num, 1) # [N, 512]
    feats_seg_3 = torch.cat([feats_seg_global, feats_seg_2], dim=-1) # [N, 512+3+3+3+512]
    pred_mask_s2_1 = model.seg_s2_mlp_1(feats_seg_3).squeeze(-1) # [N]
    pred_mask_s2_2 = model.seg_s2_mlp_2(feats_seg_3).squeeze(-1) # [N]
    pred_mask_s2_3 = model.seg_s2_mlp_3(feats_seg_3).squeeze(-1) # [N]
    pred_mask_s2 = torch.stack([pred_mask_s2_1, pred_mask_s2_2, pred_mask_s2_3], dim=-1) # [N, 3]


    mask_1 = torch.sigmoid(pred_mask_s2_1)
    mask_2 = torch.sigmoid(pred_mask_s2_2)
    mask_3 = torch.sigmoid(pred_mask_s2_3)

    mask_1 = mask_1.detach().cpu().numpy() > 0.5
    mask_2 = mask_2.detach().cpu().numpy() > 0.5
    mask_3 = mask_3.detach().cpu().numpy() > 0.5

    print(feats_seg.shape, pred_mask.shape)
    feats_iou = torch.cat([feats_seg_global, feats_seg, pred_mask_s2], dim=-1) # [N, 512+3+3+3+512]
    feats_iou = model.iou_mlp(feats_iou) # [N, 512]
    feats_iou = torch.max(feats_iou, dim=0).values # [512]
    pred_iou = model.iou_mlp_out(feats_iou) # [3]
    pred_iou = torch.sigmoid(pred_iou) # [3]
    org_iou = pred_iou.detach().cpu().numpy() # [3]
    org_iou_1 = org_iou[0].item()
    org_iou_2 = org_iou[1].item()
    org_iou_3 = org_iou[2].item()
    pred_iou_1 = org_iou_1
    pred_iou_2 = org_iou_2
    pred_iou_3 = org_iou_3

    return mask_1, mask_2, mask_3, pred_iou_1, pred_iou_2, pred_iou_3, org_iou_1, org_iou_2, org_iou_3

def mask2color(mask):
    point_num = mask.shape[0]
    colors = np.expand_dims(POINT_COLOR, axis=0)
    colors = np.tile(colors, (point_num, 1))
    colors[mask] = MASK_COLOR
    return colors

def main(args):
    # load model
    print("加载模型")
    model = P3SAM()
    model.load_state_dict(args.ckpt_path)
    model.eval()
    model.cuda()  
    print("模型加载完成")

    print("加载数据列表")
    data_list = os.listdir(args.data_dir)
    print(f"共加载{len(data_list)}个数据")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")

    if args.data_dir is None:
        if args.data_id in data_list:
            data_list.remove(args.data_id)
        data_list.insert(0, args.data_id)
    cur_data_id = [data_list[0]]

    points = [None]
    points_handle = [None]
    colors_pca = [None]
    feats = [None]
    show_colors = [None]
    point_prompt = [None]
    mask_res = [None, None, None]
    iou_res = [None, None, None]
    iou_org = [None, None, None]
    best = [None]

    def remove_point_prompt():
        if point_prompt[0] is not None:
            server.scene.remove_by_name(f"/sphere")
            point_prompt[0] = None

    def clear_state():
        mask_res[0] = None
        mask_res[1] = None
        mask_res[2] = None
        iou_res[0] = None
        iou_res[1] = None
        iou_res[2] = None
        iou_org[0] = None
        iou_org[1] = None
        iou_org[2] = None
        best[0] = None
        remove_point_prompt()

    def load_pc(use_normal=True, noise_std=0):
        clear_state()

        print("加载数据")
        if args.data_dir is not None:
            glb_data_path = os.path.join(args.data_dir, cur_data_id[0])
        else:
            glb_data_path = os.path.join(args.data_root, cur_data_id[0], 'pure_mesh.glb')   

        if glb_data_path.endswith('.glb') or glb_data_path.endswith('.obj'):
            mesh = trimesh.load(glb_data_path, force='mesh', process=False)
            _points, face_idx = trimesh.sample.sample_surface(mesh, args.point_num)    
            _points = normalize_pc(_points)
            _points = _points + np.random.normal(0, 1, size=_points.shape) * noise_std
            normals = mesh.face_normals[face_idx]
            if not use_normal or args.no_normal:
                normals = normals * 0
        else:
            raise ValueError(f"Unsupported file type: {glb_data_path}")

        show_color = np.array([POINT_COLOR])
        _show_colors = np.tile(show_color, (_points.shape[0], 1))

        print("预处理特征")
        _feats = get_feat(model, _points, normals)

        print("PCA获取特征颜色")
        feat_save = _feats.float().detach().cpu().numpy()
        data_scaled = feat_save / np.linalg.norm(feat_save, axis=-1, keepdims=True)
        pca = PCA(n_components=3)
        data_reduced = pca.fit_transform(data_scaled)
        data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
        _colors_pca = (data_reduced * 255).astype(np.uint8)

        # add point cloud
        _points_handle = server.scene.add_point_cloud(
            name="/point_cloud",
            points=_points,
            colors=_show_colors, 
            point_size=POINT_SIZE,
        )
        points[0] = _points
        points_handle[0] = _points_handle
        colors_pca[0] = _colors_pca
        feats[0] = _feats
        show_colors[0] = _show_colors
        print("加载数据完成")

    load_pc()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:

        data_list_handle = client.gui.add_dropdown(
            "Data ID", data_list
        )

        click_button_handle = client.gui.add_button(
            "Add Point Prompt", icon=viser.Icon.POINTER
        )

        clear_button_handle = client.gui.add_button(
            "Clear Point Prompt", icon=viser.Icon.X
        )

        markdown_handle = client.gui.add_markdown(
            "IOU: 1: 0.000, 2: 0.000, 3: 0.000"
        )

        drop_down_handle = client.gui.add_dropdown(
            "Segmentation", ["Mask-1", "Mask-2", "Mask-3"]
        )

        checkbox_handle = client.gui.add_checkbox(
            "Show Feature", initial_value=False
        )

        checkbox_handle_2 = client.gui.add_checkbox(
            "use normal", initial_value=True
        )

        slider_handle = client.gui.add_slider(
            "Point Size",
            min=0.00025,
            max=0.005,
            step=0.00025,
            initial_value=0.001,
        )

        slider_handle_2 = client.gui.add_slider(
            "Point Noise",
            min=0,
            max=0.02,
            step=0.0005,
            initial_value=0,
        )

        def show_mask():
            if not checkbox_handle.value:
                mask_name = drop_down_handle.value
                flag = False 
                if mask_name == "Mask-1":
                    if mask_res[0] is not None:
                        points_handle[0].colors = mask2color(mask_res[0])
                        flag = True
                elif mask_name == "Mask-2":
                    if mask_res[1] is not None:
                        points_handle[0].colors = mask2color(mask_res[1])
                        flag = True
                elif mask_name == "Mask-3":
                    if mask_res[2] is not None:
                        points_handle[0].colors = mask2color(mask_res[2])
                        flag = True

                if iou_res[0] is not None:
                    text = "IOU: "
                    for i in range(3):
                        if best[0] == i:
                            text += f"<font color=\"red\">{i+1}: {iou_res[i]:.3f}</font> "
                        else:
                            text += f"{i+1}: {iou_res[i]:.3f} "
                    text += '\n\n'
                    text += "Org IOU: "
                    for i in range(3):
                        text += f"{i+1}: {iou_org[i]:.3f} "
                    markdown_handle.content = text
                else:
                    markdown_handle.content = "IOU: 1: 0.000, 2: 0.000, 3: 0.000"

                if not flag:
                    points_handle[0].colors = show_colors[0]
            else:
                points_handle[0].colors = colors_pca[0]

        def add_point_prompt():
            if point_prompt[0] is None:
                return
            select_point = point_prompt[0]
            server.scene.add_icosphere(
                name=f"/sphere",
                radius=0.01,
                color=PROMPT_COLOR,
                position=select_point,
            )
        

        @click_button_handle.on_click
        def _(_):
            click_button_handle.disabled = True

            @client.scene.on_pointer_event(event_type="click")
            def _(event: viser.ScenePointerEvent) -> None:
                o = np.array(event.ray_origin)
                d = np.array(event.ray_direction)
                
                A = points[0] - o 
                B = np.expand_dims(d, axis=0)
                AB = np.sum(A * B, axis=-1)
                B_squre = np.sum(B ** 2, axis=-1)
                t = AB / B_squre
                intersect_points = o + t.reshape(-1, 1) * d
                distv = np.sum((intersect_points - points[0]) ** 2, axis=-1) ** 0.5
                disth = t*np.sqrt(B_squre)
                mask = (distv < POINT_SIZE)
                if np.sum(mask) == 0:
                    mask = (distv < POINT_SIZE*5)
                    if np.sum(mask) == 0:
                        return
                select_points = points[0][mask]
                disth = disth[mask]
                min_disth_idx = np.argmin(disth)
                select_point = select_points[min_disth_idx]
                print(f"选择点: {select_point}")
                point_prompt[0] = select_point
                add_point_prompt()

                pred_mask_1, pred_mask_2, pred_mask_3, pred_iou_1, pred_iou_2, pred_iou_3, org_iou_1, org_iou_2, org_iou_3 = get_mask(model, feats[0], points[0], point_prompt[0])
                mask_res[0] = pred_mask_1
                mask_res[1] = pred_mask_2
                mask_res[2] = pred_mask_3
                iou_res[0] = pred_iou_1
                iou_res[1] = pred_iou_2
                iou_res[2] = pred_iou_3
                iou_org[0] = org_iou_1
                iou_org[1] = org_iou_2
                iou_org[2] = org_iou_3
                best[0] = np.argmax(np.array([pred_iou_1, pred_iou_2, pred_iou_3]))
                if best[0] == 0:
                    mask_name = "Mask-1"
                elif best[0] == 1:
                    mask_name = "Mask-2"
                elif best[0] == 2:
                    mask_name = "Mask-3"
                drop_down_handle.value = mask_name

                print('获取mask成功', np.sum(mask_res[0]), np.sum(mask_res[1]), np.sum(mask_res[2]))
                print('获取iou成功', pred_iou_1, pred_iou_2, pred_iou_3)
                print('最佳mask', best[0]+1)

                client.scene.remove_pointer_callback()

            @client.scene.on_pointer_callback_removed
            def _():
                click_button_handle.disabled = False

        @drop_down_handle.on_update
        def _(_):
            show_mask()

        @checkbox_handle.on_update
        def _(_):
            show_mask()
        
        @ checkbox_handle_2.on_update
        def _(_):
            load_pc(checkbox_handle_2.value, slider_handle_2.value)
            show_mask()
        
        @clear_button_handle.on_click
        def _(_):
            clear_state()
            show_mask()

        @slider_handle.on_update
        def _(_):
            global POINT_SIZE
            points_handle[0].point_size = slider_handle.value
            POINT_SIZE = slider_handle.value
        
        @slider_handle_2.on_update
        def _(_):
            load_pc(checkbox_handle_2.value, slider_handle_2.value)
            show_mask()
        
        @data_list_handle.on_update
        def _(_):
            cur_data_id[0] = data_list_handle.value
            load_pc(checkbox_handle_2.value)
            show_mask()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ckpt_path', type=str, default=None, help='path to continue ckpt')
    argparser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    argparser.add_argument("--port", default=8080, type=int, help="Port to bind to")
    argparser.add_argument("--point_num", default=100000, type=int, help="Number of points to sample from the mesh")
    argparser.add_argument("--data_dir", default='../assets', type=str, help="Data directory")
    argparser.add_argument("--no_normal", action='store_true', help="Do not use normal information")
    args = argparser.parse_args()

    main(args)

'''
python app.py
python app.py --ckpt_path ../weights/p3sam.ckpt --data_dir ../assets
'''