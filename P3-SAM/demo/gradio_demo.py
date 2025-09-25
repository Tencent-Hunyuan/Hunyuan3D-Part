import gradio as gr
import os
import sys
import argparse
import numpy as np

from auto_mask import AutoMask
from auto_mask_no_postprocess import AutoMask as AutoMaskNoPostProcess

import trimesh

argparser = argparse.ArgumentParser()
argparser.add_argument('--ckpt_path', type=str, default=None, help='模型路径')
args = argparser.parse_args()

automask = AutoMask(args.ckpt_path)
automask_no_postprocess = AutoMaskNoPostProcess(args.ckpt_path, automask_instance=automask)

def load_mesh(mesh_file_name, post_process, seed):
    mesh = trimesh.load(mesh_file_name, force='mesh', process=False)
    if post_process:
        aabb, face_ids, mesh = automask.predict_aabb(mesh, seed=seed, is_parallel=False, post_process=False)
    else:
        aabb, face_ids, mesh = automask_no_postprocess.predict_aabb(mesh, seed=seed, is_parallel=False, post_process=False)
    color_map = {}
    unique_ids = np.unique(face_ids)
    for i in unique_ids:
        if i == -1:
            continue
        part_color = np.random.rand(3) * 255
        color_map[i] = part_color
    face_colors = []
    for i in face_ids:
        if i == -1:
            face_colors.append([0, 0, 0])
        else:
            face_colors.append(color_map[i])
    face_colors = np.array(face_colors).astype(np.uint8)
    mesh_save = mesh.copy()
    mesh_save.visual.face_colors = face_colors

    file_path = 'segment_result.glb'
    mesh_save.export(file_path)
    return file_path


demo = gr.Interface(
    description=
'''
## P3-SAM: Native 3D Part Segmentation

[Paper](https://arxiv.org/abs/2509.06784) | [Project Page](https://murcherful.github.io/P3-SAM/) | [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/P3-SAM/) | [Model](https://huggingface.co/tencent/Hunyuan3D-Part)

This is a demo of P3-SAM, a native 3D part segmentation method that can segment a mesh into different parts.
Input a mesh and push the "submit" button to get the segmentation results.
''',
    fn=load_mesh,
    inputs=[
        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Input Mesh"), 
        gr.Checkbox(value=True, label="Connectivty"),
        gr.Number(value=42, label="Random Seed", )],
    outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Segmentation Results"),
    examples=[
        [os.path.join(os.path.dirname(__file__), "assets/1.glb")],
        [os.path.join(os.path.dirname(__file__), "assets/2.glb")],
        [os.path.join(os.path.dirname(__file__), "assets/3.glb")],
        [os.path.join(os.path.dirname(__file__), "assets/4.glb")],
    ],
    flagging_mode='never',
)


if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=8080)

'''
python gradio_demo.py 
python gradio_demo.py --ckpt_path ../weights/p3sam.ckpt
'''