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

import torch
from .part_encoders import PartEncoder
from ..autoencoders import VolumeDecoderShapeVAE
from ...utils.misc import (
    instantiate_from_config,
    instantiate_non_trainable_model,
)
from .sonata_extractor import SonataFeatureExtractor
from .part_encoders import PartEncoder


def debug_sonata_feat(points, feats):
    from sklearn.decomposition import PCA
    import numpy as np
    import trimesh
    import os

    point_num = points.shape[0]
    feat_save = feats.float().detach().cpu().numpy()
    data_scaled = feat_save / np.linalg.norm(feat_save, axis=-1, keepdims=True)
    pca = PCA(n_components=3)
    data_reduced = pca.fit_transform(data_scaled)
    data_reduced = (data_reduced - data_reduced.min()) / (
        data_reduced.max() - data_reduced.min()
    )
    colors_255 = (data_reduced * 255).astype(np.uint8)
    colors_255 = np.concatenate(
        [colors_255, np.ones((point_num, 1), dtype=np.uint8) * 255], axis=-1
    )
    pc_save = trimesh.points.PointCloud(points, colors=colors_255)
    return pc_save
    # pc_save.export(os.path.join("debug", "point_pca.glb"))


class Conditioner(torch.nn.Module):

    def __init__(
        self,
        use_image=False,
        use_geo=True,
        use_obj=True,
        use_seg_feat=False,
        geo_cfg=None,
        obj_encoder_cfg=None,
        seg_feat_cfg=None,
        **kwargs
    ):
        super().__init__()
        self.use_image = use_image
        self.use_obj = use_obj
        self.use_geo = use_geo
        self.use_seg_feat = use_seg_feat
        self.geo_cfg = geo_cfg
        self.obj_encoder_cfg = obj_encoder_cfg
        self.seg_feat_cfg = seg_feat_cfg
        if use_geo and geo_cfg is not None:
            self.geo_encoder: PartEncoder = instantiate_from_config(geo_cfg)
            if hasattr(geo_cfg, "output_dim"):
                self.geo_out_proj = torch.nn.Linear(1024 + 512, geo_cfg.output_dim)

        if use_obj and obj_encoder_cfg is not None:
            self.obj_encoder: VolumeDecoderShapeVAE = instantiate_non_trainable_model(
                obj_encoder_cfg
            )
            if hasattr(obj_encoder_cfg, "output_dim"):
                self.obj_out_proj = torch.nn.Linear(
                    1024 + 512, obj_encoder_cfg.output_dim
                )
        if use_seg_feat and seg_feat_cfg is not None:
            self.seg_feat_encoder: SonataFeatureExtractor = (
                instantiate_non_trainable_model(seg_feat_cfg)
            )
            if hasattr(seg_feat_cfg, "output_dim"):
                self.seg_feat_outproj = torch.nn.Linear(512, seg_feat_cfg.output_dim)

    def forward(self, part_surface_inbbox, object_surface):
        bz = part_surface_inbbox.shape[0]
        context = {}
        # geo_cond
        if self.use_geo:
            context["geo_cond"], local_pc_infos = self.geo_encoder(
                part_surface_inbbox,
                object_surface,
                return_local_pc_info=True,
            )
        # obj cond
        if self.use_obj:
            with torch.no_grad():
                context["obj_cond"], global_pc_infos = self.obj_encoder.encode_shape(
                    object_surface, return_pc_info=True
                )

        # seg feat cond
        if self.use_seg_feat:
            # TODO: batchsize must be One
            num_parts = part_surface_inbbox.shape[0]
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                # encode sonata feature
                # with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    point, normal = (
                        object_surface[:1, ..., :3].float(),
                        object_surface[:1, ..., 3:6].float(),
                    )
                    point_feat = self.seg_feat_encoder(point, normal)
            # local feat
            if self.use_obj:
                nearest_global_matches = torch.argmin(
                    torch.cdist(global_pc_infos[0], object_surface[..., :3]), dim=-1
                )
                # global feat
                global_point_feats = point_feat.expand(num_parts, -1, -1).gather(
                    1,
                    nearest_global_matches.unsqueeze(-1).expand(
                        -1, -1, point_feat.size(-1)
                    ),
                )
                context["obj_cond"] = torch.concat(
                    [context["obj_cond"], global_point_feats], dim=-1
                ).to(dtype=self.obj_out_proj.weight.dtype)
                if hasattr(self, "obj_out_proj"):
                    context["obj_cond"] = self.obj_out_proj(
                        context["obj_cond"]
                    )  # .float()
            if self.use_geo:
                nearest_local_matches = torch.argmin(
                    torch.cdist(local_pc_infos[0], object_surface[..., :3]), dim=-1
                )
                local_point_feats = point_feat.expand(num_parts, -1, -1).gather(
                    1,
                    nearest_local_matches.unsqueeze(-1).expand(
                        -1, -1, point_feat.size(-1)
                    ),
                )
                context["geo_cond"] = torch.concat(
                    [context["geo_cond"], local_point_feats],
                    dim=-1,
                ).to(dtype=self.geo_out_proj.weight.dtype)
                if hasattr(self, "geo_out_proj"):
                    context["geo_cond"] = self.geo_out_proj(
                        context["geo_cond"]
                    )  # .float()
        return context
