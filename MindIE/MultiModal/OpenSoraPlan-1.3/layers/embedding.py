#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from einops import rearrange
import torch_npu

from .activation import get_activation_fn


def get_embedding_helper(embedding_type: str, embdding_dim: int):
    match embedding_type:
        case None:
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported embedding_type:{embedding_type}.")


class PatchEmbed2D(nn.Module):

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )

    def forward(self, latent):
        b, _, _, _, _ = latent.shape
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)
        latent = rearrange(latent, '(b t) c h w -> b (t h w) c', b=b)
        return latent
    

class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        return conditioning
    

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        #num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb
    

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):

    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,  # 256
        time_embed_dim: int, # embedding_dim
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation_fn(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation_fn(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

    
class RoPE3D(torch.nn.Module):

    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        # for (ntokens x batch_size x nheads x dim)
        cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]
        # equal to  (tokens * cos) + (rotate_half(tokens) * sin)
        return rotary_mul(tokens, cos, sin)
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: ntokens x batch_size x nheads x dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (ntokens x batch_size x nheads x dim)
        """
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2# Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens
    

class PositionGetter3D():
    """ return positions of patches """

    def __init__(self, ):
        self.cache_positions = {}
        
    def __call__(self, b, t, h, w, device):
        if not (b,t,h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]

        return pos
    

def rotary_mul(tokens, cos, sin):
    return torch_npu.npu_rotary_mul(tokens, cos, sin)


class AdaLayerNormSingle(nn.Module):

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep