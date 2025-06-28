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

import importlib
import torch
import torch.nn as nn
import numpy as np


def rearrange(x: torch.Tensor, datatype, t, s):
    rearrange_function = {
        'B (T S) C -> (B S) T C': rearrange_b_ts_c_2_bs_t_c,
        '(B S) T C -> B (T S) C': rearrange_bs_t_c_2_b_ts_c,
        'B (T S) C -> (B T) S C': rearrange_b_ts_c_2_bt_s_c,
        '(B T) S C -> B (T S) C': rearrange_bt_s_c_2_b_ts_c,
        'B (T S) C -> B T S C': rearrange_b_ts_c_2_b_t_s_c,
        'B T S C -> B (T S) C': rearrange_b_t_s_c_2_b_ts_c,
    }
    rearrange_func = rearrange_function.get(datatype, None)
    if rearrange_func is None:
        raise ValueError(f"Unsupported rearrange type: {datatype}")
    return rearrange_func(x, t, s)


def rearrange_b_ts_c_2_bs_t_c(x, t, s):
    shape = x.shape
    x = x.view(shape[0], t, s, shape[-1])
    x = x.transpose(1, 2)
    return x.reshape(shape[0] * s, t, shape[-1])


def rearrange_bs_t_c_2_b_ts_c(x, t, s):
    shape = x.shape
    x = x.view(-1, s, t, shape[-1])
    x = x.transpose(1, 2)
    return x.reshape(-1, t * s, shape[-1])


def rearrange_b_ts_c_2_bt_s_c(x, t, s):
    shape = x.shape
    return x.reshape(-1, s, shape[-1])


def rearrange_bt_s_c_2_b_ts_c(x, t, s):
    shape = x.shape
    return x.reshape(-1, t * s, shape[-1])


def rearrange_b_ts_c_2_b_t_s_c(x, t, s):
    shape = x.shape
    return x.reshape(shape[0], t, s, shape[-1])


def rearrange_b_t_s_c_2_b_ts_c(x, t, s):
    shape = x.shape
    return x.reshape(shape[0], t * s, shape[-1])


def rearrange_flatten_t(x):
    x_shape = x.shape
    x = x.transpose(1, 2)
    return x.view((x_shape[0] * x_shape[2]), x_shape[1], x_shape[3], x_shape[4])


def rearrange_unflatten_t(x, b):
    x_shape = x.shape
    x = x.view(b, x_shape[0] // b, x_shape[1], x_shape[2], x_shape[3])
    return x.transpose(1, 2)


def get_2d_sincos_pos_embed(
        embed_dim, grid_size, cls_token=False, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    extra_tokens = 0
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(
        embed_dim, length, interpolation_scale=1.0, base_size=16
):
    pos = torch.arange(0, length).unsqueeze(1) / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb