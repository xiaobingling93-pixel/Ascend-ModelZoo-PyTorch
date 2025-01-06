# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import math
from typing import Optional

import torch
from torch import nn
from diffusers.models.activations import get_activation


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    max_period: int = 10000,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    interpolation_scale=1.0,
    base_size=16,
):
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        torch.arange(grid_size[0], dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`torch.Tensor`): Grid of positions with shape `(H * W,)`.

    Returns:
        `torch.Tensor`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`torch.Tensor`): 1D tensor of positions with shape `(M,)`

    Returns:
        `torch.Tensor`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
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

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

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


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CogView3CombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, pooled_projection_dim: int, timesteps_dim: int = 256):
        super().__init__()

        self.time_proj = Timesteps(num_channels=timesteps_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.condition_proj = Timesteps(num_channels=condition_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=timesteps_dim, time_embed_dim=embedding_dim)
        self.condition_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(
        self,
        timestep: torch.Tensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)

        original_size_proj = self.condition_proj(original_size.flatten()).view(original_size.size(0), -1)
        crop_coords_proj = self.condition_proj(crop_coords.flatten()).view(crop_coords.size(0), -1)
        target_size_proj = self.condition_proj(target_size.flatten()).view(target_size.size(0), -1)

        condition_proj = torch.cat([original_size_proj, crop_coords_proj, target_size_proj], dim=1)

        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (B, embedding_dim)
        condition_emb = self.condition_embedder(condition_proj.to(dtype=hidden_dtype))  # (B, embedding_dim)

        conditioning = timesteps_emb + condition_emb
        return conditioning


class CogView3PlusPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        text_hidden_size: int = 4096,
        pos_embed_max_size: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.text_hidden_size = text_hidden_size
        self.pos_embed_max_size = pos_embed_max_size
        # Linear projection for image patches
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

        # Linear projection for text embeddings
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)

        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, pos_embed_max_size, base_size=pos_embed_max_size
        )
        pos_embed = pos_embed.reshape(pos_embed_max_size, pos_embed_max_size, hidden_size)
        self.register_buffer("pos_embed", pos_embed.float(), persistent=False)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Height and width must be divisible by patch size")

        height = height // self.patch_size
        width = width // self.patch_size
        hidden_states = hidden_states.view(batch_size, channel, height, self.patch_size, width, self.patch_size)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).contiguous()
        hidden_states = hidden_states.view(batch_size, height * width, channel * self.patch_size * self.patch_size)

        # Project the patches
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Calculate text_length
        text_length = encoder_hidden_states.shape[1]

        image_pos_embed = self.pos_embed[:height, :width].reshape(height * width, -1)
        text_pos_embed = torch.zeros(
            (text_length, self.hidden_size), dtype=image_pos_embed.dtype, device=image_pos_embed.device
        )
        pos_embed = torch.cat([text_pos_embed, image_pos_embed], dim=0)[None, ...]

        return (hidden_states + pos_embed).to(hidden_states.dtype)