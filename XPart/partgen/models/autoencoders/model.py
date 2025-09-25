# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import os
from typing import Tuple, List, Union

from functools import partial

import copy
import numpy as np
import torch
import torch.nn as nn
import yaml

from .attention_blocks import (
    FourierEmbedder,
    Transformer,
    CrossAttentionDecoder,
    PointCrossAttentionEncoder,
)
from .surface_extractors import MCSurfaceExtractor, SurfaceExtractors, Latent2MeshOutput
from .volume_decoders import (
    VanillaVolumeDecoder,
)
from ...utils.misc import logger, synchronize_timer, smart_load_model
from ...utils.mesh_utils import extract_geometry_fast


class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        deterministic=False,
        feat_dim=1,
    ):
        """
        Initialize a diagonal Gaussian distribution with mean and log-variance parameters.

        Args:
            parameters (Union[torch.Tensor, List[torch.Tensor]]):
                Either a single tensor containing concatenated mean and log-variance along `feat_dim`,
                or a list of two tensors [mean, logvar].
            deterministic (bool, optional): If True, the distribution is deterministic (zero variance).
                Default is False. feat_dim (int, optional): Dimension along which mean and logvar are
                concatenated if parameters is a single tensor. Default is 1.
        """
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        """
        Sample from the diagonal Gaussian distribution.

        Returns:
            torch.Tensor: A sample tensor with the same shape as the mean.
        """
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2, 3)):
        """
        Compute the Kullback-Leibler (KL) divergence between this distribution and another.

        If `other` is None, compute KL divergence to a standard normal distribution N(0, I).

        Args:
            other (DiagonalGaussianDistribution, optional): Another diagonal Gaussian distribution.
            dims (tuple, optional): Dimensions along which to compute the mean KL divergence.
                Default is (1, 2, 3).

        Returns:
            torch.Tensor: The mean KL divergence value.
        """
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class VectsetVAE(nn.Module):

    @classmethod
    @synchronize_timer("VectsetVAE Model Loading")
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device="cuda",
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace(".ckpt", ".safetensors")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        logger.info(f"Loading model from {ckpt_path}")
        if use_safetensors:
            import safetensors.torch

            ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        model_kwargs = config["params"]
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt)
        model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device="cuda",
        dtype=torch.float16,
        use_safetensors=False,
        variant="fp16",
        subfolder="hunyuan3d-vae-v2-1",
        **kwargs,
    ):
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and"
            f" {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def __init__(self, volume_decoder=None, surface_extractor=None):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        with synchronize_timer("Volume decoding"):
            grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        with synchronize_timer("Surface extraction"):
            outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs


class VolumeDecoderShapeVAE(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_encoder_layers: int = 8,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
        use_ln_post: bool = True,
        ckpt_path=None,
        volume_decoder=None,
        surface_extractor=None,
    ):
        super().__init__(volume_decoder, surface_extractor)
        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs, include_pi=include_pi
        )

        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            downsample_ratio=self.downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            qk_norm=qk_norm,
        )

        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def encode(self, surface, sample_posterior=True, return_pc_info=False):
        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents, pc_infos = self.encoder(pc, feats)
        # print(latents.shape, self.pre_kl.weight.shape)
        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            latents = posterior.sample()
        else:
            latents = posterior.mode()
        if return_pc_info:
            return latents, pc_infos
        else:
            return latents

    def encode_shape(self, surface, return_pc_info=False):
        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents, pc_infos = self.encoder(pc, feats)
        if return_pc_info:
            return latents, pc_infos
        else:
            return latents

    def decode(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries=queries, latents=latents).squeeze(-1)
        return logits

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        coarse_kwargs = copy.deepcopy(kwargs)
        coarse_kwargs["octree_resolution"] = 256

        with synchronize_timer("Coarse Volume decoding"):
            coarse_grid_logits = self.volume_decoder(
                latents, self.geo_decoder, **coarse_kwargs
            )
        with synchronize_timer("Coarse Surface extraction"):
            coarse_mesh = self.surface_extractor(coarse_grid_logits, **coarse_kwargs)

        assert len(coarse_mesh) == 1
        bbox_gen_by_coarse_matching_cube_mesh = np.stack(
            [coarse_mesh[0].mesh_v.max(0), coarse_mesh[0].mesh_v.min(0)]
        )
        bbox_gen_by_coarse_matching_cube_mesh_range = (
            bbox_gen_by_coarse_matching_cube_mesh[0]
            - bbox_gen_by_coarse_matching_cube_mesh[1]
        )

        # extend by 10%
        bbox_gen_by_coarse_matching_cube_mesh[0] += (
            bbox_gen_by_coarse_matching_cube_mesh_range * 0.1
        )
        bbox_gen_by_coarse_matching_cube_mesh[1] -= (
            bbox_gen_by_coarse_matching_cube_mesh_range * 0.1
        )
        with synchronize_timer("Fine-grained Volume decoding"):
            grid_logits = self.volume_decoder(
                latents,
                self.geo_decoder,
                bbox_corner=bbox_gen_by_coarse_matching_cube_mesh[None],
                **kwargs,
            )
        with synchronize_timer("Fine-grained Surface extraction"):
            outputs = self.surface_extractor(
                grid_logits,
                bbox_corner=bbox_gen_by_coarse_matching_cube_mesh[None],
                **kwargs,
            )

        return outputs

    def latent2mesh_2(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = 1.1,
        octree_depth: int = 7,
        num_chunks: int = 10000,
        mc_level: float = -1 / 512,
        octree_resolution: int = None,
        mc_mode: str = "mc",
    ) -> List[Latent2MeshOutput]:
        """
        Args:
            latents: [bs, num_latents, dim]
            bounds:
            octree_depth:
            num_chunks:
        Returns:
            mesh_outputs (List[MeshOutput]): the mesh outputs list.
        """
        outputs = []
        geometric_func = partial(self.query_geometry, latents=latents)
        # 2. decode geometry
        device = latents.device
        if mc_mode == "dmc" and not hasattr(self, "diffdmc"):
            from diso import DiffDMC

            self.diffdmc = DiffDMC(dtype=torch.float32).to(device)
        mesh_v_f, has_surface = extract_geometry_fast(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=False,
            mc_level=mc_level,
            octree_resolution=octree_resolution,
            diffdmc=self.diffdmc if mc_mode == "dmc" else None,
            mc_mode=mc_mode,
        )
        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue
            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            outputs.append(out)
        return outputs
