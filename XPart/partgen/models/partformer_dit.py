# Newest version: add local&global context (cross-attn), and local&global attn (self-attn)
import math

import torch.nn.functional as F

import torch.nn as nn
import torch
from typing import Optional
from einops import rearrange
from .moe_layers import MoEBlock
import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        downscale_freq_shift: float = 0.0,
        scale: int = 1,
        max_period: int = 10000,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        cond_proj_dim=None,
        out_size=None,
    ):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(
                cond_proj_dim, frequency_embedding_size, bias=False
            )

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):

        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        # t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)

        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = (
                norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight
        # zero init
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim :, :]
            kv_dca = self.kv_proj_dca(context_dca).view(
                b, self.dca_dim, 2, self.num_heads, self.head_dim
            )
            k_dca, v_dca = kv_dca.unbind(dim=2)  # [b, s, h, d]
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, : (token_len - self.dca_dim), :]

        _, s2, c = y.shape  # [b, s2, 1024]
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        k = k.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]
        v = v.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]

        q = self.q_norm(q)
        k = self.k_norm(k)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            q, k, v = map(
                lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads),
                (q, k, v),
            )
            context = (
                F.scaled_dot_product_attention(q, k, v)
                .transpose(1, 2)
                .reshape(b, s1, -1)
            )

        if self.with_dca:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                k_dca, v_dca = map(
                    lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads),
                    (k_dca, v_dca),
                )
                context_dca = (
                    F.scaled_dot_product_attention(q, k_dca, v_dca)
                    .transpose(1, 2)
                    .reshape(b, s1, -1)
                )

            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)  # context.reshape - B, L1, -1

        return out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        use_global_processor=False,
    ):
        super().__init__()
        self.use_global_processor = use_global_processor
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(dim, dim)

        # set processor
        self.processor = LocalGlobalProcessor(use_global=use_global_processor)

    def forward(self, x):
        return self.processor(self, x)


class AttentionPool(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)

        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class LocalGlobalProcessor:
    def __init__(self, use_global=False):
        self.use_global = use_global

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
    ):
        """
        hidden_states: [B, L, C]
        """
        if self.use_global:
            B_old, N_old, C_old = hidden_states.shape
            hidden_states = hidden_states.reshape(1, -1, C_old)
        B, N, C = hidden_states.shape

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // attn.num_heads // 3
        qkv = qkv.view(1, -1, attn.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, attn.num_heads, attn.head_dim).transpose(
            1, 2
        )  # [b, h, s, d]
        k = k.reshape(B, N, attn.num_heads, attn.head_dim).transpose(
            1, 2
        )  # [b, h, s, d]
        v = v.reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)

        q = attn.q_norm(q)  # [b, h, s, d]
        k = attn.k_norm(k)  # [b, h, s, d]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            hidden_states = F.scaled_dot_product_attention(q, k, v)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, N, -1)

        hidden_states = attn.out_proj(hidden_states)
        if self.use_global:
            hidden_states = hidden_states.reshape(B_old, N_old, -1)
        return hidden_states


class PartFormerDitBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        use_self_attention: bool = True,
        use_cross_attention: bool = False,
        use_cross_attention_2: bool = False,
        encoder_hidden_dim=1024,  # cross-attn encoder_hidden_states  dim
        encoder_hidden2_dim=1024,  # cross-attn 2 encoder_hidden_states  dim
        # cross_attn2_weight=0.0,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        skip_connection=False,
        timested_modulate=False,
        c_emb_size=0,  # time embedding size
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        # self.cross_attn2_weight = cross_attn2_weight
        use_ele_affine = True
        # ========================= Self-Attention =========================
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            self.norm1 = norm_layer(
                hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
            )
            self.attn1 = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=qk_norm_layer,
            )

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(c_emb_size, hidden_size, bias=True)
            )
        # ========================= Cross-Attention =========================
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.norm2 = norm_layer(
                hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
            )
            self.attn2 = CrossAttention(
                hidden_size,
                encoder_hidden_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=qk_norm_layer,
                with_decoupled_ca=False,
            )
        self.use_cross_attention_2 = use_cross_attention_2
        if self.use_cross_attention_2:
            self.norm2_2 = norm_layer(
                hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
            )
            self.attn2_2 = CrossAttention(
                hidden_size,
                encoder_hidden2_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=qk_norm_layer,
                with_decoupled_ca=with_decoupled_ca,
                decoupled_ca_dim=decoupled_ca_dim,
                decoupled_ca_weight=decoupled_ca_weight,
            )
        # ========================= FFN =========================
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.use_moe = use_moe
        if self.use_moe:
            print("using moe")
            self.moe = MoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
                ff_inner_dim=int(hidden_size * 4.0),
                ff_bias=True,
            )
        else:
            self.mlp = MLP(width=hidden_size)
        # ========================= skip FFN =========================
        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        skip_value: torch.Tensor = None,
    ):
        # skip connection
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, hidden_states], dim=-1)
            hidden_states = self.skip_linear(cat)
            hidden_states = self.skip_norm(hidden_states)
        # local global attn (self-attn)
        if self.timested_modulate:
            shift_msa = self.default_modulation(temb).unsqueeze(dim=1)
            hidden_states = hidden_states + shift_msa
        if self.use_self_attention:
            attn_output = self.attn1(self.norm1(hidden_states))
            hidden_states = hidden_states + attn_output
        # image cross attn
        if self.use_cross_attention:
            original_cross_out = self.attn2(
                self.norm2(hidden_states),
                encoder_hidden_states,
            )
        # added local-global cross attn
        # 2. Cross-Attention
        if self.use_cross_attention_2:
            cross_out_2 = self.attn2_2(
                self.norm2_2(hidden_states),
                encoder_hidden_states_2,
            )
        hidden_states = (
            hidden_states
            + (original_cross_out if self.use_cross_attention else 0)
            + (cross_out_2 if self.use_cross_attention_2 else 0)
        )

        # FFN Layer
        mlp_inputs = self.norm3(hidden_states)

        if self.use_moe:
            hidden_states = hidden_states + self.moe(mlp_inputs)
        else:
            hidden_states = hidden_states + self.mlp(mlp_inputs)

        return hidden_states


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(
            final_hidden_size, elementwise_affine=True, eps=1e-6
        )
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x


class PartFormerDITPlain(nn.Module):

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        use_self_attention=True,
        use_cross_attention=True,
        use_cross_attention_2=True,
        encoder_hidden_dim=1024,  # cross-attn encoder_hidden_states  dim
        encoder_hidden2_dim=1024,  # cross-attn 2 encoder_hidden_states  dim
        depth=24,
        num_heads=16,
        qk_norm=False,
        qkv_bias=True,
        norm_type="layer",
        qk_norm_type="rms",
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        # use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == "layer" else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == "rms" else nn.LayerNorm
        # embedding
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(
            hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim
        )
        # Will use fixed sin-cos embedding:
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, input_size, hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # self.use_attention_pooling = use_attention_pooling
        # if use_attention_pooling:

        #     self.pooler = AttentionPool(
        #         self.text_len, encoder_hidden_dim, num_heads=8, output_dim=1024
        #     )
        #     self.extra_embedder = nn.Sequential(
        #         nn.Linear(1024, hidden_size * 4),
        #         nn.SiLU(),
        #         nn.Linear(hidden_size * 4, hidden_size, bias=True),
        #     )
        # for part embedding
        self.use_bbox_cond = kwargs.get("use_bbox_cond", False)
        if self.use_bbox_cond:
            self.bbox_conditioner = BboxEmbedder(
                out_size=hidden_size,
                num_freqs=kwargs.get("num_freqs", 8),
            )
        self.use_part_embed = kwargs.get("use_part_embed", False)
        if self.use_part_embed:
            self.valid_num = kwargs.get("valid_num", 50)
            self.part_embed = nn.Parameter(torch.randn(self.valid_num, hidden_size))
            # zero init part_embed
            self.part_embed.data.zero_()
        # transformer blocks
        self.blocks = nn.ModuleList([
            PartFormerDitBlock(
                hidden_size,
                num_heads,
                use_self_attention=use_self_attention,
                use_cross_attention=use_cross_attention,
                use_cross_attention_2=use_cross_attention_2,
                encoder_hidden_dim=encoder_hidden_dim,  # cross-attn encoder_hidden_states  dim
                encoder_hidden2_dim=encoder_hidden2_dim,  # cross-attn 2 encoder_hidden_states  dim
                # cross_attn2_weight=cross_attn2_weight,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=self.norm,
                qk_norm_layer=self.qk_norm,
                with_decoupled_ca=with_decoupled_ca,
                decoupled_ca_dim=decoupled_ca_dim,
                decoupled_ca_weight=decoupled_ca_weight,
                skip_connection=layer > depth // 2,
                use_moe=True if depth - layer <= num_moe_layers else False,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            for layer in range(depth)
        ])
        # set local-global processor
        for layer, block in enumerate(self.blocks):
            if hasattr(block, "attn1") and (layer + 1) % 2 == 0:
                block.attn1.processor = LocalGlobalProcessor(use_global=True)

        self.depth = depth

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, contexts: dict, **kwargs):
        """

        x: [B, N, C]
        t: [B]
        contexts: dict
            image_context: [B, K*ni, C]
            geo_context: [B, K*ng, C] or [B, K*ng, C*2]
        aabb: [B, K, 2, 3]
        num_tokens: [B, N]

        N = K * num_tokens

        For parts pretrain : K = 1
        """
        #  prepare input
        aabb: torch.Tensor = kwargs.get("aabb", None)
        # image_context = contexts.get("image_un_cond", None)
        object_context = contexts.get("obj_cond", None)
        geo_context = contexts.get("geo_cond", None)
        num_tokens: torch.Tensor = kwargs.get("num_tokens", None)
        # timeembedding and input projection
        t = self.t_embedder(t, condition=kwargs.get("guidance_cond"))
        x = self.x_embedder(x)

        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(x.dtype)
            x = x + pos_embed

        # c is time embedding (adding pooling context or not)
        # if self.use_attention_pooling:
        #     # TODO: attention_pooling for all contexts
        #     extra_vec = self.pooler(image_context, None)
        #     c = t + self.extra_embedder(extra_vec)  # [B, D]
        # else:
        #     c = t
        c = t
        # bounding box
        if self.use_bbox_cond:
            center_extent = torch.cat(
                [torch.mean(aabb, dim=-2), aabb[..., 1, :] - aabb[..., 0, :]], dim=-1
            )
            bbox_embeds = self.bbox_conditioner(center_extent)
            # TODO: now only support batch_size=1
            bbox_embeds = torch.repeat_interleave(
                bbox_embeds, repeats=num_tokens[0], dim=1
            )
            x = x + bbox_embeds
        # part id embedding
        if self.use_part_embed:
            num_parts = aabb.shape[1]
            random_idx = torch.randperm(self.valid_num)[:num_parts]
            part_embeds = self.part_embed[random_idx].unsqueeze(1)
            # import pdb

            # pdb.set_trace()
            x = x + part_embeds
        x = torch.cat([c, x], dim=1)
        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(
                hidden_states=x,
                # encoder_hidden_states=image_context,
                encoder_hidden_states=object_context,
                encoder_hidden_states_2=geo_context,
                temb=c,
                skip_value=skip_value,
            )
            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)
        return x
