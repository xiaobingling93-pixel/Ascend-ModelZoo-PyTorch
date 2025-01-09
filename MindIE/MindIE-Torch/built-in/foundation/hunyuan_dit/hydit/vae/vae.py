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

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple

import torch
import torch.nn as nn

from mindiesd import ConfigMixin
from ..models.model_utils import DiffusionModel
from ..layers.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block, Blockconfig


class AutoencoderKLConfig(ConfigMixin):
    config_name = 'config.json'

    def __init__(
            self,
            in_channels: int = 3, 
            out_channels: int = 3, 
            down_block_types: Tuple[str] = ("DownEncoderBlock2D",), 
            up_block_types: Tuple[str] = ("UpDecoderBlock2D",), 
            block_out_channels: Tuple[int] = (64,), 
            layers_per_block: int = 1, 
            act_fn: str = "silu", 
            latent_channels: int = 4, 
            norm_num_groups: int = 32, 
            sample_size: int = 32, 
            scaling_factor: float = 0.18215, 
            shift_factor: Optional[float] = None, 
            latents_mean: Optional[Tuple[float]] = None, 
            latents_std: Optional[Tuple[float]] = None,  
            force_upcast: float = True,  
            use_quant_conv: bool = True, 
            use_post_quant_conv: bool = True, 
        ):
        super().__init__()  

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.latent_channels = latent_channels
        self.norm_num_groups = norm_num_groups
        self.sample_size = sample_size
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.force_upcast = force_upcast
        self.use_quant_conv = use_quant_conv
        self.use_post_quant_conv = use_post_quant_conv


class AutoencoderKL(DiffusionModel):

    config_class = AutoencoderKLConfig

    def __init__(self, config: AutoencoderKLConfig):
        super().__init__(config)

        # Initialize Encoder with config parameters
        self.encoder = Encoder2D(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=config.down_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
            double_z=True,
        )

        # Initialize Decoder with config parameters
        self.decoder = Decoder2D(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
        )
        
        self.quant_conv = None
        self.post_quant_conv = None
        if config.use_quant_conv:
            self.quant_conv = nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        if config.use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)

        # Tiling configuration
        self.tile_sample_min_size = config.sample_size
        sample_size = config.sample_size[0] if isinstance(config.sample_size, (list, tuple)) else config.sample_size
        self.tile_latent_min_size = int(sample_size / (2 ** (len(config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def encode(self, x: torch.Tensor):
    
        h = self.encoder(x)
        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h
        posterior = DiagonalGaussianDistribution(moments)

        return (posterior,)

    def decode(self, z: torch.FloatTensor):

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        decoded = self.decoder(z)

        return (decoded,)

    def _load_weights(self, state_dict):
        weights = state_dict
        # attention_block:
        for i in ["encoder", "decoder"]:
            prefix_key = i + '.mid_block.attentions.0.'
            to_q_weight = weights.pop(prefix_key + 'to_q.weight')
            to_q_bias = weights.pop(prefix_key + 'to_q.bias')
            to_k_weight = weights.pop(prefix_key + 'to_k.weight')
            to_k_bias = weights.pop(prefix_key + 'to_k.bias')
            to_v_weight = weights.pop(prefix_key + 'to_v.weight')
            to_v_bias = weights.pop(prefix_key + 'to_v.bias')
            weights[prefix_key + 'qkv_proj.weight'] = torch.cat(
                [to_q_weight, to_k_weight, to_v_weight], dim=0).transpose(0, 1).contiguous()
            weights[prefix_key + 'qkv_proj.bias'] = torch.cat([to_q_bias, to_k_bias, to_v_bias], dim=0)
            weights[prefix_key + 'out_proj.weight'] = weights.pop(prefix_key + 'to_out.0.weight')
            weights[prefix_key + 'out_proj.bias'] = weights.pop(prefix_key + 'to_out.0.bias')
        self.load_state_dict(weights)


class Decoder2D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            blockconfig = Blockconfig(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            up_block = get_up_block(up_block_type, blockconfig)
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            raise ValueError("Now not support norm type==`spatial`")
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Encoder2D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            blockconfig = Blockconfig(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            down_block = get_down_block(down_block_type, blockconfig)
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DiagonalGaussianDistribution(object):
    def __init__(
            self,
            parameters,
            deterministic=False,
    ):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device, dtype=self.mean.dtype)

    def sample(self):
        # torch.randn: standard normal distribution
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype=self.mean.dtype)
        return x

    def mode(self):
        return self.mean