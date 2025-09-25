import torch.nn as nn
from ...utils.misc import (
    instantiate_from_config,
    instantiate_non_trainable_model,
)
from ..autoencoders.model import (
    VolumeDecoderShapeVAE,
)


class PartEncoder(nn.Module):
    def __init__(
        self,
        use_local=True,
        local_global_feat_dim=None,
        local_geo_cfg=None,
        local_feat_type="latents",
        num_tokens_cond=2048,
    ):
        super().__init__()
        self.local_global_feat_dim = local_global_feat_dim
        self.local_feat_type = local_feat_type
        self.num_tokens_cond = num_tokens_cond
        # local
        self.use_local = use_local
        if use_local:
            if local_geo_cfg is None:
                raise ValueError(
                    "local_geo_cfg must be provided when use_local is True"
                )
            assert (
                "ShapeVAE" in local_geo_cfg.get("target").split(".")[-1]
            ), "local_geo_cfg must be a ShapeVAE config"
            self.local_encoder: VolumeDecoderShapeVAE = instantiate_from_config(
                local_geo_cfg
            )
            if self.local_global_feat_dim is not None:
                self.local_out_layer = nn.Linear(
                    (
                        local_geo_cfg.params.embed_dim
                        if self.local_feat_type == "latents"
                        else local_geo_cfg.params.width
                    ),
                    self.local_global_feat_dim,
                    bias=True,
                )

    def forward(self, part_surface_inbbox, object_surface, return_local_pc_info=False):
        """
        Args:
            aabb: (B, 2, 3) tensor representing the axis-aligned bounding box
            object_surface: (B, N, 3) tensor representing the surface points of the object
        Returns:
            local_features: (B, num_tokens_cond, C) tensor of local features
            global_features: (B,num_tokens_cond, C) tensor of global features
        """
        # random selection if more than num_tokens_cond points
        if self.use_local:
            # with torch.autocast(
            #     device_type=part_surface_inbbox.device.type,
            #     dtype=torch.float16,
            # ):
            # with torch.no_grad():
            if self.local_feat_type == "latents":
                local_features, local_pc_infos = self.local_encoder.encode(
                    part_surface_inbbox, sample_posterior=True, return_pc_info=True
                )  # (B, num_tokens_cond, C)
            elif self.local_feat_type == "latents_shape":
                local_features, local_pc_infos = self.local_encoder.encode_shape(
                    part_surface_inbbox, return_pc_info=True
                )  # (B, num_tokens_cond, C)
            elif self.local_feat_type == "miche-point-query-structural-vae":
                local_features, local_pc_infos = self.local_encoder.encode(
                    part_surface_inbbox, sample_posterior=True, return_pc_info=True
                )
                local_features = self.local_encoder(local_features)
            else:
                raise ValueError(
                    f"local_feat_type {self.local_feat_type} not supported"
                )
            # ouput layer
            geo_features = (
                self.local_out_layer(local_features)
                if hasattr(self, "local_out_layer")
                else local_features
            )
        if return_local_pc_info:
            return geo_features, local_pc_infos
        return geo_features
