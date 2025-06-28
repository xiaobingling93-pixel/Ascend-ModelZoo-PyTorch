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

import os
import logging
from typing import Tuple, Optional, Union
import glob
import importlib
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import pytorch_lightning as pl
from diffusers.configuration_utils import register_to_config
from diffusers import ModelMixin, ConfigMixin
from opensoraplan.layers import nonlinearity, normalize
from opensoraplan.utils.utils import path_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Module = str
MODULES_BASE = "opensoraplan.layers."
SPATIAL_DOWNSAMPLE = "Downsample"
RESNET_BLOCK_2D = "ResnetBlock2D"
RESNET_BLOCK_3D = "ResnetBlock3D"
SPATIAL_UPSAMPLE_2X = "SpatialUpsample2x"
SPATIAL_DOWNSAMPLE_2X = "SpatialDownsample2x"
TIME_DOWNSAMPLE_2X = "TimeDownsample2x"
CAUSAL_CONV_3D = "CausalConv3d"
LATENTS_SCALING_FACTOR = 0.18215


def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class VideoBaseAePl(pl.LightningModule, ModelMixin, ConfigMixin):
    config_name = "config.json"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        ckpt_files = glob.glob(os.path.join(pretrained_model_name_or_path, '*.ckpt'))
        if ckpt_files:
            # Adapt to PyTorch Lightning
            last_ckpt_file = ckpt_files[-1]
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            model = cls.from_config(config_file)
            logger.info("init from %s", last_ckpt_file)
            model.init_from_ckpt(last_ckpt_file)
            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample):
        dims = [1, 2, 3]
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class Encoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "CasualConv3d",
        attention: Module = "AttnBlock",
        resnet_blocks: Tuple[Module] = (
            RESNET_BLOCK_2D,
            RESNET_BLOCK_2D,
            RESNET_BLOCK_2D,
            RESNET_BLOCK_3D,
        ),
        spatial_downsample: Tuple[Module] = (
            SPATIAL_DOWNSAMPLE,
            SPATIAL_DOWNSAMPLE,
            SPATIAL_DOWNSAMPLE,
            "",
        ),
        temporal_downsample: Tuple[Module] = ("", "", "TimeDownsampleRes2x", ""),
        mid_resnet: Module = RESNET_BLOCK_3D,
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        if len(resnet_blocks) != len(hidden_size_mult):
            logger.error("resnet_blocks size does not equal to hidden_size_mult size.")
            raise ValueError
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        self.conv_in = resolve_str_to_obj(conv_in)(
            3, hidden_size, kernel_size=3, stride=1, padding=1
        )

        # ---- Downsample ----
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]
            block_out = hidden_size * hidden_size_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_obj(attention)(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if spatial_downsample[i_level]:
                down.downsample = resolve_str_to_obj(spatial_downsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res // 2
            if temporal_downsample[i_level]:
                down.time_downsample = resolve_str_to_obj(temporal_downsample[i_level])(
                    block_in, block_in
                )
            self.down.append(down)

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = resolve_str_to_obj(attention)(block_in)
        self.mid.block_2 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        # ---- Out ----
        self.norm_out = normalize(block_in)
        self.conv_out = resolve_str_to_obj(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
            if hasattr(self.down[i_level], "time_downsample"):
                hs_down = self.down[i_level].time_downsample(hs[-1])
                hs.append(hs_down)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "CasualConv3d",
        attention: Module = "AttnBlock",
        resnet_blocks: Tuple[Module] = (
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
        ),
        spatial_upsample: Tuple[Module] = (
            "",
            SPATIAL_UPSAMPLE_2X,
            SPATIAL_UPSAMPLE_2X,
            SPATIAL_UPSAMPLE_2X,
        ),
        temporal_upsample: Tuple[Module] = ("", "", "", "TimeUpsampleRes2x"),
        mid_resnet: Module = RESNET_BLOCK_3D,
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = resolve_str_to_obj(conv_in)(
            z_channels, block_in, kernel_size=3, padding=1
        )

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = resolve_str_to_obj(attention)(block_in)
        self.mid.block_2 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # ---- Upsample ----
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_obj(attention)(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if spatial_upsample[i_level]:
                up.upsample = resolve_str_to_obj(spatial_upsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res * 2
            if temporal_upsample[i_level]:
                up.time_upsample = resolve_str_to_obj(temporal_upsample[i_level])(
                    block_in, block_in
                )
            self.up.insert(0, up)

        # ---- Out ----
        self.norm_out = normalize(block_in)
        self.conv_out = resolve_str_to_obj(conv_out)(
            block_in, 3, kernel_size=3, padding=1
        )

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
            if hasattr(self.up[i_level], "time_upsample"):
                h = self.up[i_level].time_upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class CausalVAEModel(VideoBaseAePl):
    @register_to_config
    def __init__(
        self,
        lr: float = 1e-5,
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = None,
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        q_conv: str = CAUSAL_CONV_3D,
        encoder_conv_in: Module = CAUSAL_CONV_3D,
        encoder_conv_out: Module = CAUSAL_CONV_3D,
        encoder_attention: Module = "AttnBlock3D",
        encoder_resnet_blocks: Tuple[Module] = (
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
        ),
        encoder_spatial_downsample: Tuple[Module] = (
            SPATIAL_DOWNSAMPLE_2X,
            SPATIAL_DOWNSAMPLE_2X,
            SPATIAL_DOWNSAMPLE_2X,
            "",
        ),
        encoder_temporal_downsample: Tuple[Module] = (
            "",
            TIME_DOWNSAMPLE_2X,
            TIME_DOWNSAMPLE_2X,
            "",
        ),
        encoder_mid_resnet: Module = RESNET_BLOCK_3D,
        decoder_conv_in: Module = CAUSAL_CONV_3D,
        decoder_conv_out: Module = CAUSAL_CONV_3D,
        decoder_attention: Module = "AttnBlock3D",
        decoder_resnet_blocks: Tuple[Module] = (
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
            RESNET_BLOCK_3D,
        ),
        decoder_spatial_upsample: Tuple[Module] = (
            "",
            SPATIAL_UPSAMPLE_2X,
            SPATIAL_UPSAMPLE_2X,
            SPATIAL_UPSAMPLE_2X,
        ),
        decoder_temporal_upsample: Tuple[Module] = ("", "", "TimeUpsample2x", "TimeUpsample2x"),
        decoder_mid_resnet: Module = RESNET_BLOCK_3D,
    ) -> None:
        super().__init__()
        self.tile_sample_min_size = 256
        self.tile_sample_min_size_t = 65
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(hidden_size_mult) - 1)))
        self.tile_overlap_factor = 0.25
        self.use_tiling = False

        self.learning_rate = lr
        self.lr_g_factor = 1.0

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
        )

        quant_conv_cls = resolve_str_to_obj(q_conv)
        self.quant_conv = quant_conv_cls(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)
        self.patch_size = (1, 8, 8)

    def encode(self, x):
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size
            or x.shape[-2] > self.tile_sample_min_size
        ):
            return self.tiled_encode2d(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.tiled_decode2d(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, model_inputs, sample_posterior=True):
        posterior = self.encode(model_inputs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        alphas = torch.linspace(0, 1, blend_extent, device=a.device).view(-1, 1).expand(-1, a.shape[4])
        b[:, :, :, :blend_extent, :] = (
                    a[:, :, :, -blend_extent:, :] * (1 - alphas) + b[:, :, :, :blend_extent, :] * alphas)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        alphas = torch.linspace(0, 1, blend_extent, device=a.device).expand(a.shape[3], -1)
        b[:, :, :, :, :blend_extent] = (
                    a[:, :, :, :, -blend_extent:] * (1 - alphas) + b[:, :, :, :, :blend_extent] * alphas)
        return b

    def tiled_encode2d(self, x):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i: i + self.tile_sample_min_size,
                    j: j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def tiled_decode2d(self, z):

        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i: i + self.tile_latent_min_size,
                    j: j + self.tile_latent_min_size,
                ]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def init_from_ckpt(self, path, ignore_keys, remove_loss=True):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    logger.error("Deleting key %s from state_dict.", k)
                    del sd[k]
            if remove_loss and "loss" in k:
                del sd[k]
        self.load_state_dict(sd, strict=False)


class CausalVAEModelWrapper(nn.Module):
    def __init__(self, vae, latent_size):
        super(CausalVAEModelWrapper, self).__init__()
        self.vae = vae
        self.latent_size = latent_size

    @classmethod
    def from_pretrained(cls, model_path, latent_size, cache_dir, **kwargs):
        real_path = path_check(model_path)
        if len(latent_size) != 2 or latent_size[0] <= 0 or latent_size[1] <= 0:
            raise ValueError("latent_size shape or value is invalid.")
        causal_vae = CausalVAEModel.from_pretrained(real_path, cache_dir=cache_dir, **kwargs)
        return cls(causal_vae, latent_size)

    def decode(self, x):
        x = self.vae.decode(x / LATENTS_SCALING_FACTOR)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x
