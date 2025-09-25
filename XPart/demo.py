# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from partgen.partformer_pipeline import PartFormerPipeline

import pytorch_lightning as pl
from partgen.utils.misc import get_config_from_file
import argparse
import numpy as np
import torch


def main(ckpt_path, config, mesh_path, save_dir, ignore_keys=()):
    # Set random seed
    pl.seed_everything(2026, workers=True)
    # pipeline = PartFormerPipeline.from_single_file(
    #     ckpt_path=ckpt_path,
    #     config=config,
    #     verbose=True,
    #     ignore_keys=ignore_keys,
    # )
    pipeline = PartFormerPipeline.from_pretrained(
        config=config,
        verbose=True,
        ignore_keys=ignore_keys,
    )
    pipeline.to(device="cuda", dtype=torch.float32)
    cfg = -1.0
    # for mesh paths
    uid = Path(mesh_path).stem
    additional_params = {"output_type": "trimesh"}
    obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
        mesh_path=mesh_path,
        octree_resolution=512,
        **additional_params,
    )
    obj_mesh.export(save_dir / f"train_cfg_{cfg:04f}_dit_boxgpt_{uid}.glb")
    out_bbox.export(save_dir / f"train_cfg_{cfg:04f}_dit_boxgpt_{uid}_bbox.glb")
    mesh_gt_bbox.export(save_dir / f"train_cfg_{cfg:04f}_input_boxgpt_{uid}.glb")
    explode_object.export(save_dir / f"train_cfg_{cfg:04f}_explode_boxgpt_{uid}.glb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
    )
    # input_dir
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="./data/test.glb",
    )
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()
    config = get_config_from_file(
        args.config
        if args.config
        else str(Path(__file__).parent / "partgen/config" / "infer.yaml")
    )
    assert hasattr(config, "ckpt") or hasattr(
        config, "ckpt_path"
    ), "ckpt or ckpt_path must be specified in config"
    ckpt_path = Path(args.ckpt if args.ckpt else config.ckpt_path)
    ignore_keys = config.get("ignore_keys", [])
    save_dir = (
        Path(args.save_dir)
        / ckpt_path.parent.parent.relative_to(
            ckpt_path.parent.parent.parent.parent.parent
        )
        # / ckpt_path.stem
    )
    print(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    main(
        ckpt_path=ckpt_path,
        config=config,
        mesh_path=args.mesh_path,
        save_dir=save_dir,
        ignore_keys=ignore_keys,
    )
