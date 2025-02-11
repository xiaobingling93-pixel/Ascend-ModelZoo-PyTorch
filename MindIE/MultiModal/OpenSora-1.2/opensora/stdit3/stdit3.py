#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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
import functools
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from mindiesd.config_utils import ConfigMixin
from mindiesd.models.model_utils import DiffusionModel
from ..layer import approx_gelu                          
from ..layer import Attention, MultiHeadCrossAttention
from ..layer import CaptionEmbedder, PatchEmbed3D, PositionEmbedding2D, SizeEmbedder, \
    TimestepEmbedder, RotaryEmbedding
from ..layer import Mlp
from ..layer import AdaLayerNorm
from ..layer import (
    all_to_all_with_pad,
    get_spatial_pad,
    get_temporal_pad,
    set_spatial_pad,
    set_temporal_pad,
    split_sequence,
    gather_sequence,
)
from ..layer import get_sequence_parallel_group

MAX_IN_CHANNELS = 4
MAX_CAPTIOIN_CHANNELS = 4096


class STDiT3Config(ConfigMixin):
    config_name = 'config.json'

    def __init__(
            self,
            input_size: Tuple[int, int, int] = (None, None, None),
            in_channels: int = 4,
            caption_channels: int = 4096,
            enable_flash_attn: bool = True,
            enable_sequence_parallelism: bool = False,
            use_cache: bool = True,
            cache_interval: int = 2,
            cache_start: int = 3,
            num_cache_layer: int = 13,
            cache_start_steps: int = 5,
    ):
        super().__init__()

        self.input_size = input_size
        self.in_channels = in_channels
        self.caption_channels = caption_channels
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.use_cache = use_cache
        self.cache_interval = cache_interval
        self.cache_start = cache_start
        self.num_cache_layer = num_cache_layer
        self.cache_start_steps = cache_start_steps


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = AdaLayerNorm(hidden_size, eps=1e-6)

        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, t, s):
        # x: [B, (T, S), C], mased_x: [B, (T, S), C], x_mask: [B, T]
        x = rearrange(x, "b (t s) c -> b t s c", t=t, s=s)
        masked_x = rearrange(masked_x, "b (t s) c -> b t s c", t=t, s=s)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "b t s c -> b (t s) c")
        return x

    def forward(self, x, t, x_mask=None, t0=None, t_s=(None, None)):
        d_t = t_s[0]
        d_s = t_s[1]
        if d_t is None:
            d_t = self.d_t
        if d_s is None:
            d_s = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = self.norm_final(x, shift, 1 + scale[0])

        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = self.norm_final(x, shift_zero, 1 + scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, d_t, d_s)
        x = self.linear(x)
        return x


class STDiT3Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            rope=None,
            qk_norm=False,
            temporal=False,
            enable_flash_attn=False,
            enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        attn_cls = Attention
        mha_cls = MultiHeadCrossAttention

        self.norm1 = AdaLayerNorm(hidden_size, eps=1e-6)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )

        self.cross_attn = mha_cls(hidden_size, num_heads)
        self.norm2 = AdaLayerNorm(hidden_size, eps=1e-6)

        self.mlp = Mlp(
            features_in=hidden_size, features_hidden=int(hidden_size * mlp_ratio), act_layer=approx_gelu)
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size) / hidden_size ** 0.5)

    def t_mask_select(self, x_mask, x, masked_x, t, s):
        # x: [B, (T, S), C], mased_x: [B, (T, S), C], x_mask: [B, T]
        x = rearrange(x, "b (t s) c -> b t s c", t=t, s=s)
        masked_x = rearrange(masked_x, "b (t s) c -> b t s c", t=t, s=s)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "b t s c -> b (t s) c")
        return x

    def forward(
            self, x, y, t, mask=None, x_mask=None, t0=None, number_frames=None, number_pixel_patches=None
    ):
        # prepare modulate parameters
        x_shape0_b, x_shape1_n, x_shape2_c = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(x_shape0_b, 6, -1)
        ).chunk(6, dim=1)

        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                    self.scale_shift_table[None] + t0.reshape(x_shape0_b, 6, -1)
            ).chunk(6, dim=1)

        # modulate attention
        x_m = self.norm1(x, shift_msa, 1 + scale_msa[0])
        if x_mask is not None:
            x_m_zero = self.norm1(x, shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, number_frames, number_pixel_patches)

        # modulate attention
        if self.temporal:
            if self.enable_sequence_parallelism:
                x_m, number_pixel_patches, number_frames = self.dynamic_switch(
                    x_m, number_pixel_patches, number_frames, temporal_to_spatial=True)
            x_m = rearrange(x_m, "b (t s) c -> (b s) t c", t=number_frames, s=number_pixel_patches)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(b s) t c -> b (t s) c", t=number_frames, s=number_pixel_patches)
            # because x_mask split on the dim 1 
            if self.enable_sequence_parallelism:
                x_m, number_pixel_patches, number_frames = self.dynamic_switch(
                    x_m, number_pixel_patches, number_frames, temporal_to_spatial=False)

        else:
            x_m = rearrange(x_m, "b (t s) c -> (b t) s c", t=number_frames, s=number_pixel_patches)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(b t) s c -> b (t s) c", t=number_frames, s=number_pixel_patches)

        # modulate attention
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, number_frames, number_pixel_patches)

        # residual
        x = x + x_m_s

        # cross attention
        x = x + self.cross_attn(x, y, mask)

        # modulate MLP
        x_m = self.norm2(x, shift_mlp, 1 + scale_mlp[0])
        if x_mask is not None:
            x_m_zero = self.norm2(x, shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, number_frames, number_pixel_patches)

        # MLP
        x_m = self.mlp(x_m)

        # modulate MLP
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, number_frames, number_pixel_patches)

        # residual
        x = x + x_m_s

        return x

    def dynamic_switch(self, x, s, t, temporal_to_spatial: bool):
        if temporal_to_spatial:
            scatter_dim, gather_dim = 2, 1
            scatter_pad = get_spatial_pad()
            gather_pad = get_temporal_pad()
        else:
            scatter_dim, gather_dim = 1, 2
            scatter_pad = get_temporal_pad()
            gather_pad = get_spatial_pad()

        x = rearrange(x, "b (t s) c -> b t s c", t=t, s=s)

        x = all_to_all_with_pad(
            x,
            get_sequence_parallel_group(),
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            scatter_pad=scatter_pad,
            gather_pad=gather_pad,
        )

        new_s, new_t = x.shape[2], x.shape[1]

        x = rearrange(x, "b t s c -> b (t s) c", t=new_t, s=new_s)
        return x, new_s, new_t


class STDiT3(DiffusionModel):
    config_class = STDiT3Config
    weigths_name = 'model.safetensors'

    def __init__(self, config):
        super().__init__(config)

        self.pred_sigma = True
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if self.pred_sigma else config.in_channels

        # model size related
        self.depth = 28
        self.mlp_ratio = 4.0
        self.hidden_size = 1152
        self.num_heads = 16

        # computation related
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_sequence_parallelism = config.enable_sequence_parallelism

        # input size related
        self.patch_size = (1, 2, 2)
        self.input_sq_size = 512
        self.pos_embed = PositionEmbedding2D(self.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self._init_embedding(config)

        self._init_blocks(config)

        # final layer
        self.final_layer = T2IFinalLayer(self.hidden_size, np.prod(self.patch_size), self.out_channels)

        self._initialize_weights()

        self._init_cache(config)

    def forward(
        self, 
        x: torch.Tensor, 
        timestep: torch.Tensor, 
        y: torch.Tensor, 
        mask: torch.Tensor = None, 
        x_mask: torch.Tensor = None, 
        fps: torch.Tensor = None, 
        height: torch.Tensor = None, 
        width: torch.Tensor = None, 
        t_idx: int = 0, 
        **kwargs
    ) -> torch.Tensor:

        dtype = self.x_embedder.proj.weight.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        if fps is None:
            fps = torch.tensor([8], device=x.device)
        if height is None:
            height = torch.tensor([720], device=x.device)
        if width is None:
            width = torch.tensor([1280], device=x.device)

        # get shape
        _, _, tx, hx, wx = x.size()
        x_shape0_t, x_shape1_h, x_shape2_w = self._get_dynamic_size(x)
        s_hw = x_shape1_h * x_shape2_w
        resolution_sq = (height[0].to(torch.float32).item() * width[0].to(torch.float32).item()) ** 0.5
        scale = resolution_sq / self.input_sq_size

        # === get pos embed ===
        pos_emb = self.pos_embed(x, x_shape1_h, x_shape2_w, scale)

        # === get timestep embed ===
        t, t_mlp, t0, t0_mlp = self._get_t_embed(x, timestep, fps, x_mask)

        # === get y embed ===
        y, y_lens = self._get_y_embed(y, mask)

        # === get x embed ===
        x = self._get_x_embed(x, pos_emb, x_shape0_t, s_hw)

        x = rearrange(x, "b t s c -> b (t s) c", t=x_shape0_t, s=s_hw)

        x = self._forward_blocks(x, y, x_mask, x_shape0_t, s_hw, y_lens, t_mlp, t0_mlp, t_idx)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, (x_shape0_t, s_hw))
        x = self._unpatchify(x, x_shape0_t, x_shape1_h, x_shape2_w, (tx, hx, wx))

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def _init_embedding(self, config):
        self.x_embedder = PatchEmbed3D(self.patch_size, config.in_channels, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=self.hidden_size,
            act_layer=approx_gelu,
            token_num=300,
        )

    def _init_blocks(self, config):
        # spatial blocks
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_norm=True,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for _ in range(self.depth)
            ]
        )

        # temporal blocks
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_norm=True,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                )
                for _ in range(self.depth)
            ]
        )

    def _init_cache(self, config):
        self.use_cache = config.use_cache
        self.cache_interval = config.cache_interval
        self.cache_start = config.cache_start
        self.num_cache_layer = config.num_cache_layer
        self.cache_start_steps = config.cache_start_steps

        self.delta_cache = None

    def _initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def _get_dynamic_size(self, x):
        _, _, x_shape2_t, x_shape3_h, x_shape4_w = x.size()
        if x_shape2_t % self.patch_size[0] != 0:
            x_shape2_t += self.patch_size[0] - x_shape2_t % self.patch_size[0]
        if x_shape3_h % self.patch_size[1] != 0:
            x_shape3_h += self.patch_size[1] - x_shape3_h % self.patch_size[1]
        if x_shape4_w % self.patch_size[2] != 0:
            x_shape4_w += self.patch_size[2] - x_shape4_w % self.patch_size[2]
        x_shape2_t = x_shape2_t // self.patch_size[0]
        x_shape3_h = x_shape3_h // self.patch_size[1]
        x_shape4_w = x_shape4_w // self.patch_size[2]
        return x_shape2_t, x_shape3_h, x_shape4_w

    def _get_t_embed(self, x, timestep, fps, x_mask):
        x_shape0_b = x.size(0)
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), x_shape0_b)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)
        return t, t_mlp, t0, t0_mlp

    def _get_x_embed(self, x, pos_emb, t, s):
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "b (t s) c -> b t s c", t=t, s=s)
        x = x + pos_emb
        return x

    def _get_y_embed(self, y, mask):
        y, y_lens = self._encode_text(y, mask)

        return y, y_lens

    def _encode_text(self, y, mask=None):
        y = self.y_embedder(y)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    # forward blocks in range [start_idx, end_idx), then return input and output
    def _forward_blocks_range(self, x, y, x_mask, t, s, y_lens, t_mlp, t0_mlp, start_idx, end_idx):
        for spatial_block, temporal_block in zip(self.spatial_blocks[start_idx: end_idx],
                                                 self.temporal_blocks[start_idx: end_idx]):
            x = spatial_block(x, y, t_mlp, y_lens, x_mask, t0_mlp, t, s)
            x = temporal_block(x, y, t_mlp, y_lens, x_mask, t0_mlp, t, s)

        return x

    def _forward_blocks(self, x, y, x_mask, t, s, y_lens, t_mlp, t0_mlp, t_idx):
        # === if dsp parallel, split x across T ===
        if self.enable_sequence_parallelism:
            set_temporal_pad(t)
            set_spatial_pad(s)
            x = rearrange(x, "b (t s) c -> b t s c", t=t, s=s)
            x = split_sequence(
                x, get_sequence_parallel_group(), dim=1, pad=get_temporal_pad()
            )
            t = x.shape[1]
            x = rearrange(x, "b t s c -> b (t s) c", t=t, s=s)
            if x_mask is not None:
                x_mask = split_sequence(
                    x_mask, get_sequence_parallel_group(), dim=1, pad=get_temporal_pad()
                )

        num_blocks = len(self.spatial_blocks)
        if not self.use_cache or (t_idx < self.cache_start_steps):
            x = self._forward_blocks_range(x, y, x_mask, t, s, y_lens, t_mlp, t0_mlp,
                                           0, num_blocks)
        else:
            # infer [0, cache_start)
            x = self._forward_blocks_range(x, y, x_mask, t, s, y_lens, t_mlp, t0_mlp,
                                           0, self.cache_start)
            # infer [cache_start, cache_end)
            cache_end = np.minimum(self.cache_start + self.num_cache_layer, num_blocks)
            x_before_cache = x.clone()
            if t_idx % self.cache_interval == (self.cache_start_steps % 2):
                x = self._forward_blocks_range(x, y, x_mask, t, s, y_lens, t_mlp, t0_mlp,
                                               self.cache_start, cache_end)
                self.delta_cache = x - x_before_cache
            else:
                x = x_before_cache + self.delta_cache
            # infer [cache_end, num_blocks)
            x = self._forward_blocks_range(x, y, x_mask, t, s, y_lens, t_mlp, t0_mlp,
                                           cache_end, num_blocks)

        # === if dsp parallel, gather x across T ===
        if self.enable_sequence_parallelism:
            x = rearrange(x, "b (t s) c -> b t s c", t=t, s=s)
            x = gather_sequence(
                x, get_sequence_parallel_group(), dim=1, pad=get_temporal_pad()
            )
            t, s = x.shape[1], x.shape[2]
            x = rearrange(x, "b t s c -> b (t s) c", t=t, s=s)

        return x

    def _unpatchify(self, x, n_t, n_h, n_w, shape_org):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        r_t, r_h, r_w = shape_org
        t_p, h_p, w_p = self.patch_size

        x_shape0_b = x.shape[0]
        x = x.reshape(x_shape0_b, n_t, n_h, n_w, t_p, h_p, w_p, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(x_shape0_b, self.out_channels, n_t * t_p, n_h * h_p, n_w * w_p)

        # unpad
        x = x[:, :, :r_t, :r_h, :r_w]
        return x
