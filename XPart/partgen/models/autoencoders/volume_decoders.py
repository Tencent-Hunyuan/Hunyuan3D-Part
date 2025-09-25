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

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm

from .attention_blocks import CrossAttentionDecoder
from ...utils.misc import logger
from ...utils.mesh_utils import (
    extract_near_surface_volume_fn,
    generate_dense_grid_points,
)


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):

        """
        Perform volume decoding with a vanilla decoder
        Args:
            latents (torch.FloatTensor): Latent vectors to decode.
            geo_decoder (Callable): The geometry decoder function.
            bounds (Union[Tuple[float], List[float], float]): Bounding box for the volume.
            num_chunks (int): Number of chunks to process at a time.
            octree_resolution (int): Resolution of the octree for sampling points.
            enable_pbar (bool): Whether to enable progress bar.
        Returns:    
            grid_logits (torch.FloatTensor): Decoded 3D volume logits.
        """
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij",
        )
        xyz_samples = (
            torch.from_numpy(xyz_samples)
            .to(device, dtype=dtype)
            .contiguous()
            .reshape(-1, 3)
        )

        # 2. latents to 3d volume
        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"Volume Decoding",
            disable=not enable_pbar,
        ):
            chunk_queries = xyz_samples[start : start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits
