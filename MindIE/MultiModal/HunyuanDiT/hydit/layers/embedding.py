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
from typing import Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch_npu


def get_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_type: str = "adjacent"):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        x (torch.Tensor): Query or key tensor to apply rotary embeddings. BSND or BNSD.
        cos (torch.Tensor): Precomputed cos frequency tensor for complex exponentials.
        sin (torch.Tensor): Precomputed sin frequency tensor for complex exponentials.
        rope_type (str):
            if "adjacent": rotate q to [-q_1, q_0, -q_3, q_2, ... , -q_d-1, q_d-2].
                           Could to be used for HunyuanDiT, OpenSora, Flux, CogVideox.
            if "symmetric": rotate q to [-q_d/2, -q_d/2+1, ... , -q_d-1, q_0, q_1, ... , q_d/2-1].
                            Could to be used for OpenSoraPlan, Stable Audio.
            if "symmetric_fuse": is equivalent to "symmetric" but has better performance in torch_npu.

    Returns:
        (torch.Tensor): modified query or key tensor with rotary embeddings.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"The type of input x must be torch.Tensor, but got {type(x)}.")
    if not isinstance(cos, torch.Tensor):
        raise ValueError(f"The type of input cos must be torch.Tensor, but got {type(cos)}.")
    if not isinstance(sin, torch.Tensor):
        raise ValueError(f"The type of input sin must be torch.Tensor, but got {type(sin)}.")
    if not isinstance(rope_type, str):
        raise ValueError(f"The type of input rope_type must be strings, but got {type(rope_type)}.")

    match rope_type:
        case "adjacent":
            # Used for HunyuanDiT, OpenSora, Flux, CogVideox
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
            return (x * cos + x_rotated * sin)
        case "symmetric":
            # Used for OpenSoraPlan, Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
            return (x * cos + x_rotated * sin)
        case "symmetric_fuse":
            return torch_npu.npu_rotary_mul(x, cos, sin)
        case _:
            raise ValueError(f"Unsupported rope_type: {rope_type}.")


def get_embedding_helper(embedding_type: str, embdding_dim: int):
    match embedding_type:
        case None:
            return nn.Identity()
        case 'rope':
            return RotaryPositionEmbedding(embed_dim=embdding_dim)
        case _:
            raise ValueError(f"Unsupported embedding_type:{embedding_type}.")


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
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)   # size: [dim/2], 一个指数衰减的曲线
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def cal_1d_sincos_embed(
        items: torch.Tensor,
        embed_dim: int,
        max_period: int = 10000,
        step: int = 1,
        flip: bool = False
    ):
    """
    Calculate 1d sinusoidal embeddings.
    Args:
        items (torch.Tensor): Items includes N indices. Must be a 1D tensor (N,).
        embed_dim (int): The dimension of the embeddings.
        max_period (int): Controls the minimum frequency of the embeddings.
        step (int): The step of frequences.
        flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
    Return:
        embed (torch.Tensor): An (N, embed_dim//step) tensor of item embeddings.
    """

    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError(f"Embed_dim should be a positive integer, but receive {embed_dim}.")
    if step not in [1, 2]:
        raise ValueError(f"Step must be in [1, 2], but receive {step}.")
    if embed_dim % (2 * step) != 0:
        raise ValueError(f"Embed_dim must be divisible by {2 * step}, but receive {embed_dim}.")

    half_of_dim = embed_dim // 2
    # generate frequency vectors
    freqs = torch.arange(start=0, end=half_of_dim, step=step, dtype=torch.float32, device=items.device)
    freqs = torch.exp(-math.log(max_period) * freqs / half_of_dim)  # (embed_dim//(2*step))
    # (N, 1) * (1, embed_dim//(2*step)) -> (N, embed_dim//(2*step))
    freqs = items[:, None].float() * freqs[None, :]
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    # (N, embed_dim//step)
    if flip:
        embed = torch.cat([cos, sin], dim=-1)
    else:
        embed = torch.cat([sin, cos], dim=-1)
        
    return embed


class SinCosPositionEmbed1D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        step: int = 1,
        flip: bool = False,
        max_period: int = 10000,
        cache1d: bool = True,
        size: int = 128
    ):
        """
        Create 1d sinusoidal embeddings.
        Args:
            embed_dim (int): The dimension of the embeddings.
            step (int): The step of frequences.
            flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
            max_period (int): Controls the minimum frequency of the embeddings.
            cache1d (bool): If true, use cache.
            size (int): The size of cache.
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.step = step
        self.flip = flip
        self.max_period = max_period
        self.cache1d = cache1d
        self.size = size
        if self.cache1d:
            items = torch.arange(self.size)
            # (size, embed_dim//step)
            embed = cal_1d_sincos_embed(items, self.embed_dim, self.max_period, self.step, self.flip)
            self.register_buffer("embed", embed, persistent=False)
        else:
            self.embed = None
    
    def get_1d_sincos_embed(self, items: torch.Tensor):
        """
        Calculate 1d sinusoidal embeddings.
        Args:
            items (torch.Tensor): Items includes N indices. Must be a 1D tensor (N,).
        Return:
            embed (torch.Tensor): An (N, embed_dim//step) tensor of item embeddings.
        """

        if len(items.shape) != 1:
            raise ValueError(f"Items should be a 1D tensor, but receive a {len(items.shape)}D tensor.")

        items_max = torch.max(items)
        dytpes = [torch.int, torch.long]
        if self.cache1d and items_max < self.size and items.dtype in dytpes:
            embed = self.embed[items]
        else:
            embed = cal_1d_sincos_embed(items, self.embed_dim, self.max_period, self.step, self.flip)
        
        return embed


class SinCosPositionEmbed2D(SinCosPositionEmbed1D):
    def __init__(
        self,
        embed_dim: int = 256,
        step: int = 1,
        flip: bool = False,
        max_period: int = 10000,
        cache2d: bool = True,
        grid_size: Union[Tuple[int, int], int] = (224, 224),
        base_size: Union[int, None] = None,
        interpolation_scale: float = 1.0,
        persistent = False,
    ):  
        """
        Create 2d sinusoidal embeddings.
        Args:
            embed_dim (int): The dimension of the embeddings.
            step (int): The step of frequences.
            flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
            max_period (int): Controls the minimum frequency of the embeddings.
            cache2d (bood): If true, use cache
            grid_size (Tuple[int, int] or int): The size of grid.
            base_size (int or None): The size of basic patches.
            interpolation_scale (float): The scale parameter.
            persistent (bool): If true, save the cache in dict.
        """
        
        self.embed_dim = embed_dim
        self.step = step
        self.flip = flip
        self.max_period = max_period
        self.cache2d = cache2d
        self.interpolation_scale = interpolation_scale

        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size)
        else:
            self.grid_size = grid_size
        if base_size is None:
            self.base_size = round((self.grid_size[0] * self.grid_size[1]) ** 0.5)
        else:
            self.base_size = base_size
        
        if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
            raise ValueError(f"Embed_dim should be a positive integer, but receive {self.embed_dim}.")
        if self.step not in [1, 2]:
            raise ValueError(f"Step must be in [1, 2], but receive {self.step}.")
        if self.embed_dim % (2 * self.step) != 0:
            raise ValueError(f"Embed_dim must be divisible by {2 * self.step}, but receive {self.embed_dim}.")
        
        self.dim = self.embed_dim // (2 // self.step)
        super().__init__(self.dim, self.step, self.flip, self.max_period, cache1d=False)

        if self.cache2d:
            pos_embed = self._get_2d_sincos_embed(self.grid_size, self.base_size, self.interpolation_scale)
            self.register_buffer("pos_embed", pos_embed, persistent=persistent)
        else:
            self.pos_embed = None

    def get_2d_sincos_embed(self, grid_size, base_size=None, interpolation_scale=1.0, device="cpu"):
        """
        Initialize frequences.
        Args:
            grid_size (Tuple[int, int] or int): The size of grid.
            base_size (int or None): The size of basic patches.
            interpolation_scale (float): The scale parameter.
        Return:
            emb (torch.Tensor): An (1, H*W, embed_dim) tensor of embeddings.
        """
        
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        is_shape_same = grid_size[0] == self.grid_size[0] and grid_size[1] == self.grid_size[1] \
            and base_size == self.base_size
        if self.cache2d and is_shape_same and interpolation_scale == self.interpolation_scale:
            embed = self.pos_embed
        else:
            embed = self._get_2d_sincos_embed(grid_size, base_size, interpolation_scale, device)

        return embed

    @functools.lru_cache(maxsize=512)
    def _get_2d_sincos_embed(self, grid_size, base_size, interpolation_scale, device="cpu"):
        """
        Initialize frequences.
        Args:
            grid_size (Tuple[int, int]): The size of grid.
            base_size (int or None): The size of basic patches.
            interpolation_scale (float): The scale parameter.
        Return:
            emb (torch.Tensor): An (H*W, embed_dim) tensor of embeddings.
        """

        grid_h = torch.arange(grid_size[0], dtype=torch.float32, device=device) / interpolation_scale
        grid_w = torch.arange(grid_size[1], dtype=torch.float32, device=device) / interpolation_scale

        if base_size is not None:
            grid_h *= base_size / grid_size[0]
            grid_w *= base_size / grid_size[1]

        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
        grid = torch.stack([grid_h.t().reshape(-1), grid_w.t().reshape(-1)], dim=0)  # (2, H*W)
        emb_h = self.get_1d_sincos_embed(grid[0])  # (H*W, embed_dim//2)
        emb_w = self.get_1d_sincos_embed(grid[1])  # (H*W, embed_dim//2)
        emb = torch.cat([emb_h, emb_w], dim=-1)  # (H*W, embed_dim)
        return emb


class PatchEmbed(SinCosPositionEmbed2D):
    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        """
        2D Image to Patch Embedding with support for position embedding.
        Args:
            height (int): Height of images.
            width (int): Weight of images.
            patch_size (int): The size of patches.
            in_channels (int): Number of input image channels.
            embed_dim (int): Number of linear projection output channels.
            layer_norm (bool): If true, use layernorm.
            flatten (bool): If true, flatten the latent.
            bias (bool): If true, use bias.
            interpolation_scale: Scale coefficient.
            pos_embed_type (str): The type of postion embedding.
            pos_embed_max_size: The size of max postion embedding.
        Adapted Models: SD3, HuanyuanDit
        """
        
        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.cache2d = False
        elif pos_embed_type == "sincos":
            self.cache2d = True
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

        super().__init__(
            embed_dim=embed_dim,
            step=1,
            cache2d=self.cache2d,
            grid_size=grid_size,
            base_size=self.base_size,
            interpolation_scale=self.interpolation_scale,
            persistent=True if pos_embed_max_size else False,
        )

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError(f"Parameter:`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        dtype_latent = latent.dtype
        latent = self.proj(latent.to(self.dtype))
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(dtype_latent)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            pos_embed = self.get_2d_sincos_embed(
                (height, width), 
                self.base_size,
                interpolation_scale=self.interpolation_scale,
                device=latent.device
            ).unsqueeze(0)

        return (latent + pos_embed).to(dtype_latent)


class RotaryCosSinEmbed:
    """
    RotaryCosSinEmbed get cos_sin tables of rope.
    """
    def __init__(
        self,
        embed_dim: int,
        use_real: bool = True,
        repeat_interleave_real: bool = True,
        theta: float = 10000.0,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        freqs_dtype = torch.float32,
    ):
        """
        Args:
        embed_dim (int): The embedding dimension size.
        use_real (bool): If `True`, return real part and imaginary part separately. Otherwise, return complex numbers.
        repeat_interleave_real (bool):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        linear_factor (float): Scaling factor for the context extrapolation. Defaults to 1.0. Use for `lumina`.
        ntk_factor (float): Scaling factor for the NTK-Aware RoPE. Defaults to 1.0. Use for `lumina`.
        freqs_dtype: Defaults to torch.float32. Only be torch.float64 for Flux.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.use_real = use_real
        self.repeat_interleave_real = repeat_interleave_real
        self.theta = theta
        self.linear_factor = linear_factor    # Use for lumina.
        self.ntk_factor = ntk_factor          # Use for lumina.
        self.freqs_dtype = freqs_dtype        # Flux: torch.float64


    def get_resize_crop_region_for_grid(self, src_h: int, src_w: int, base_size: int):
        """
        Get resize and crop region for grid.

        Args:
            src_h (int): The grid height of the positional embedding.
            src_w (int): The grid width of the positional embedding.
            base_size (int): The target size of resizing and cropping region for grid.

        Returns:
            Tuple[int]: The top-left and bottom-right coordinates of the crop.
        """
        if not isinstance(src_h, int):
            raise ValueError(f"The type of input src_h must be int, but got {type(src_h)}.")
        if not isinstance(src_w, int):
            raise ValueError(f"The type of input src_w must be int, but got {type(src_w)}.")
        if not isinstance(base_size, int):
            raise ValueError(f"The type of input base_size must be int, but got {type(base_size)}.")
        if src_h <= 0:
            raise ValueError(f"Input src_h must be greater than 0, but got {src_h}.")
        if src_w <= 0:
            raise ValueError(f"Input src_w must be greater than 0, but got {src_w}.")
        if base_size <= 0:
            raise ValueError(f"Input base_size must be greater than 0, but got {base_size}.")

        ratio = src_h / src_w
        # resize
        if ratio > 1:
            resize_height = base_size
            resize_width = int(round(base_size / src_h * src_w))
        else:
            resize_width = base_size
            resize_height = int(round(base_size / src_w * src_h))
        crop_top = int(round((base_size - resize_height) / 2.0))
        crop_left = int(round((base_size - resize_width) / 2.0))
        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


    def get_1d_rotary_pos_embed(self, pos: Union[np.ndarray, int]) -> torch.Tensor:
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        Args:
            pos (np.ndarray or int): Position indices for the frequency tensor. [S] or scalar.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex exponentials. [S, D/2].
        """
        if isinstance(pos, int):
            pos = torch.arange(pos)
        elif isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)  # type: ignore  # [S]
        else:
            raise ValueError(f"The type of input pos must be np.ndarray or int, but got {type(pos)}.")

        half_of_dim = self.embed_dim // 2

        theta = self.theta * self.ntk_factor
        freqs = torch.arange(start=0, end=half_of_dim, step=2, dtype=self.freqs_dtype, device=pos.device)  # [D/4]
        freqs = (1.0 / (theta ** (freqs[: (half_of_dim // 2)] / half_of_dim)) / self.linear_factor)  # [D/4]
        freqs = torch.outer(pos, freqs)  # [S, D/4]

        if self.use_real and self.repeat_interleave_real:
            # HunyuanDiT, Flux, CogVideox
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D/2]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D/2]
            return freqs_cos, freqs_sin
        elif self.use_real:
            # Stable Audio, Allegro
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [S, D/2]
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [S, D/2]
            return freqs_cos, freqs_sin
        else:
            # lumina
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/4]
            return freqs_cis


    def get_2d_rotary_pos_embed(self, grid_h: int, grid_w: int, base_size: int):
        """
        RoPE for image tokens with 2d structure.

        Args:
            grid_h (int): The grid height of the positional embedding.
            grid_w (int): The grid width of the positional embedding.
            base_size (int): The target size of resizing and cropping region for grid.

        Returns:
            torch.Tensor: positional embedding with shape (grid_size * grid_size, embed_dim/2).
        """
        if not isinstance(grid_h, int):
            raise ValueError(f"The type of input grid_h must be int, but got {type(grid_h)}.")
        if not isinstance(grid_w, int):
            raise ValueError(f"The type of input grid_w must be int, but got {type(grid_w)}.")
        if not isinstance(base_size, int):
            raise ValueError(f"The type of input base_size must be int, but got {type(base_size)}.")
        if grid_h <= 0:
            raise ValueError(f"Input grid_h must be greater than 0, but got {grid_h}.")
        if grid_w <= 0:
            raise ValueError(f"Input grid_w must be greater than 0, but got {grid_w}.")
        if base_size <= 0:
            raise ValueError(f"Input base_size must be greater than 0, but got {base_size}.")

        start, stop = self.get_resize_crop_region_for_grid(grid_h, grid_w, base_size)
        grid_h = np.linspace(start[0], stop[0], grid_h, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(start[1], stop[1], grid_w, endpoint=False, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)  # [2, W, H]

        grid = grid.reshape([2, 1, *grid.shape[1:]])
        # use half of dimensions to encode grid_h and grid_w
        emb_h = self.get_1d_rotary_pos_embed(grid[0].reshape(-1))  # (H*W, D/2) if use_real else (H*W, D/4)
        emb_w = self.get_1d_rotary_pos_embed(grid[1].reshape(-1))  # (H*W, D/2) if use_real else (H*W, D/4)

        if self.use_real:
            cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D)
            sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D)
            pos_embed = (cos, sin)
        else:
            pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)

        return pos_embed


class RotaryPositionEmbedding(RotaryCosSinEmbed, nn.Module):
    """
    RotaryPositionEmbedding apply rotary embeddings to input tensors using the given frequency tensor.
    """
    def __init__(
        self,
        embed_dim: int,
        grid_h: int = 64,
        grid_w: int = 64,
        base_size: int = 32,
        rope_type: str = "adjacent",
        use_real: bool = True,
        repeat_interleave_real: bool = True,
        theta: float = 10000.0,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        """
        Args:
        embed_dim (int): The embedding dimension size.
        grid_h (int): The grid height of the positional embedding.
        grid_w (int): The grid width of the positional embedding.
        base_size (int): The target size of resizing and cropping region for grid.
        rope_type (str):
            if "adjacent": rotate q to [-q_1, q_0, -q_3, q_2, ... , -q_d-1, q_d-2].
                             Could to be used for HunyuanDiT, Flux, CogVideox.
            if "symmetric": rotate q to [-q_d/2, -q_d/2+1, ... , -q_d-1, q_0, q_1, ... , q_d/2-1].
                              Could to be used for Stable Audio.
            if "symmetric-npu": is equivalent to "symmetric" but has better performance in torch_npu.
        use_real (bool): If `True`, return real part and imaginary part separately. Otherwise, return complex numbers.
        repeat_interleave_real (bool):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        linear_factor (float): Scaling factor for the context extrapolation. Defaults to 1.0. Use for `lumina`.
        ntk_factor (float): Scaling factor for the NTK-Aware RoPE. Defaults to 1.0. Use for `lumina`.
        """
        # check inputs
        if embed_dim % 4 != 0 or embed_dim <= 2:
            raise ValueError(f"Input embed_dim must be divisible by 4 and greater than 2, but got {embed_dim}.")
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"Input grid_size must be greater than 0, but got ({grid_h}, {grid_w}).")
        if base_size <= 0:
            raise ValueError(f"Input base_size must be greater than 0, but got {base_size}.")
        if theta <= 0.:
            raise ValueError(f"Input theta must be greater than 0, but got {theta}.")
        if linear_factor <= 0.:
            raise ValueError(f"Input linear_factor must be greater than 0, but got {linear_factor}.")
        if ntk_factor <= 0.:
            raise ValueError(f"Input ntk_factor must be greater than 0, but got {ntk_factor}.")

        self.rope_type = rope_type
        self.use_real = use_real
        super().__init__(embed_dim, use_real, repeat_interleave_real, theta, linear_factor, ntk_factor)

        self.freqs_cis_img = self.get_2d_rotary_pos_embed(grid_h, grid_w, base_size)


    def forward(self, x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None):
        """
        The input tensors are reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting
        compatibility. The resulting tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`): Query or key tensor to apply rotary embeddings. [B, H, S, D].
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        freqs_cis = freqs_cis if freqs_cis is not None else self.freqs_cis_img

        if self.use_real:
            cos, sin = freqs_cis  # [S, D]
            cos = cos[None, None].to(x.dtype)
            sin = sin[None, None].to(x.dtype)
            cos, sin = cos.to(x.device), sin.to(x.device)

            x_out = get_rotary_emb(x, cos, sin, self.rope_type)
            return x_out

        else:
            # used for lumina
            x_rotated = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
            return x_out.type_as(x)