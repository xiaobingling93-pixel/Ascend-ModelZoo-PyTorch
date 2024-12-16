#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from math import pi
from typing import Literal, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum, broadcast_tensors, Tensor
from einops import rearrange

from .mlp import Mlp
from ..utils.utils import exists, default


LANG_FREQS = 'lang'
PIXEL_FREQS = 'pixel'
CONSTANT_FREQS = 'constant'


def get_embedding_helper(embedding_type: str, embdding_dim: int):
    match embedding_type:
        case None:
            return nn.Identity()
        case 'rope':
            return RotaryEmbedding(dim=embdding_dim)
        case _:
            error_msg = "`embdding_type` is not supported!"
            raise ValueError(error_msg)

class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
            self,
            patch_size=(2, 4, 4),
            in_chans=3,
            embed_dim=96,
            norm_layer=None,
            flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, x_shape2_d, x_shape3_h, x_shape4_w = x.size()
        if x_shape4_w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - x_shape4_w % self.patch_size[2]))
        if x_shape3_h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - x_shape3_h % self.patch_size[1]))
        if x_shape2_d % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - x_shape2_d % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            x_shape2_d, x_size_3, x_sie_4 = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, x_shape2_d, x_size_3, x_sie_4)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
        b, dims = s.shape[0], s.shape[1]
        s = s.reshape(b * dims)
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = s_emb.view(b, dims, self.outdim)
        s_emb = s_emb.view(b, dims * self.outdim)
        return s_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
            self,
            in_channels,
            hidden_size,
            act_layer=nn.GELU(approximate="tanh"),
            token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            features_in=in_channels,
            features_hidden=hidden_size,
            features_out=hidden_size,
            act_layer=act_layer,
        )

        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels ** 0.5,
        )

    def forward(self, caption):
        caption = self.y_proj(caption)
        return caption


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, h: int, w: int, scale: Optional[float] = 1.0) -> torch.Tensor:
        s_hw = h * w
        base_size = round(s_hw ** 0.5)
        return self._get_cached_emb(x, h, w, scale, base_size)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
            self,
            x,
            h: int,
            w: int,
            scale: float = 1.0,
            base_size: Optional[int] = None,
    ):
        device = x.device
        dtype = x.dtype
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 dim,
                 custom_freqs: Optional[Tensor] = None,
                 freqs_for: Union[
                     Literal[LANG_FREQS],
                     Literal[PIXEL_FREQS],
                     Literal[CONSTANT_FREQS]
                 ] = LANG_FREQS,
                 theta=10000,
                 max_freq=10,
                 num_freqs=1,
                 learned_freq=False,
                 xpos_scale_base=512,
                 interpolate_factor=1.,
                 theta_rescale_factor=1.,
                 seq_before_head_dim=False,
                 cache_if_possible=True
                 ):
        super().__init__()
        
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store('dummy', torch.tensor(0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        self.interpolate_factor = interpolate_factor

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rearrange_nd_2_n1d(self, x, transform_type='n d -> n 1 d'):
        if transform_type == 'n d -> n 1 d':
            shape = x.shape
            x = x.view(shape[0], shape[1])
            return x.view(shape[0], 1, shape[1])
        return x

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, freq_seq_len=None):
        # 进入这个函数
        seq_dim = default(seq_dim, self.default_seq_dim)

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        if exists(freq_seq_len):
            seq_len = freq_seq_len

        freqs = self.forward(self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len,
                             offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return self.apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def get_axial_freqs(self, *dims):
        colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = colon

            new_axis_slice = (Ellipsis, *all_axis, colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    def rotate_half(self, x):
        shape = x.shape
        new_shape = shape[:-1] + (shape[-1] // 2, 2)
        x = x.view(new_shape)

        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        shape = x.shape
        new_shape = shape[:-2] + (shape[-1] * shape[-2],)
        x = x.view(new_shape)
        return x

    def apply_rotary_emb(self, freqs, t, start_index=0, scale=1., seq_dim=-2):
        if t.ndim == 3:
            seq_len = t.shape[seq_dim]
            freqs = freqs[-seq_len:].to(t)

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

        cos = freqs.cos() * scale
        sin = freqs.sin() * scale
        t = (t * cos) + (self.rotate_half(t) * sin)

        return torch.cat((t_left, t, t_right), dim=-1)

    def forward(
            self,
            t: Tensor,
            seq_len=None,
            offset=0
    ):
        should_cache = (
                self.cache_if_possible and \
                not self.learned_freq and \
                exists(seq_len) and \
                self.freqs_for != 'pixel'
        )

        if (
                should_cache and \
                exists(self.cached_freqs) and \
                (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs