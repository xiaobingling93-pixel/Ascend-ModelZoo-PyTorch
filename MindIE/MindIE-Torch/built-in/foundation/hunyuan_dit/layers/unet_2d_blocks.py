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


from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn

from .attention import ReconstitutionAttention
from .sampling2d import Downsample2D, Upsample2D
from .resnet import ResnetBlock2D


@dataclass
class Blockconfig:
    num_layers: int 
    in_channels: int 
    out_channels: int
    temb_channels: int
    resnet_eps: float
    resnet_act_fn: str
    add_upsample: bool = None
    add_downsample: bool = None
    num_attention_heads: Optional[int] = None
    resnet_groups: Optional[int] = None
    resnet_time_scale_shift: str = "default"
    attention_head_dim: Optional[int] = None
    downsample_padding: Optional[int] = None

    def __post_init__(self):
        # If attn head dim is not defined, we default it to the number of heads
        if self.attention_head_dim is None:
            self.attention_head_dim = self.num_attention_heads


def get_down_block(
    down_block_type: str,
    blockconfig: Blockconfig,
):
    if not isinstance(blockconfig, Blockconfig):
        raise ValueError("Please use Blockconfig to pass the parameters.")

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=blockconfig.num_layers,
            in_channels=blockconfig.in_channels,
            out_channels=blockconfig.out_channels,
            add_downsample=blockconfig.add_downsample,
            resnet_eps=blockconfig.resnet_eps,
            resnet_act_fn=blockconfig.resnet_act_fn,
            resnet_groups=blockconfig.resnet_groups,
            downsample_padding=blockconfig.downsample_padding,
            resnet_time_scale_shift=blockconfig.resnet_time_scale_shift,
        )
    else:
        raise ValueError(f"{down_block_type} does not exits.")
    

def get_up_block(
    up_block_type: str,
    blockconfig: dataclass,
) -> nn.Module:
    
    if not isinstance(blockconfig, Blockconfig):
        raise ValueError("Please use Blockconfig to pass the parameters.")
        
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=blockconfig.num_layers,
            in_channels=blockconfig.in_channels,
            out_channels=blockconfig.out_channels,
            add_upsample=blockconfig.add_upsample,
            resnet_eps=blockconfig.resnet_eps,
            resnet_act_fn=blockconfig.resnet_act_fn,
            resnet_groups=blockconfig.resnet_groups,
            resnet_time_scale_shift=blockconfig.resnet_time_scale_shift, 
            temb_channels=blockconfig.temb_channels,
        )
    else:
        raise ValueError(f"{up_block_type} does not exist.")


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(
                out_channels, use_conv=True, 
                out_channels=out_channels, padding=downsample_padding)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial  # USE deflalt
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        resnets = []
        for _ in range(num_layers + 1):
            resnets.append(ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ))

        if attention_head_dim is None:
            attention_head_dim = in_channels
        attentions = []
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    ReconstitutionAttention(
                        in_channels,
                        num_heads=in_channels // attention_head_dim,
                        head_dim=attention_head_dim,
                        eps=resnet_eps,
                        num_norm_groups=attn_groups,  
                        qkv_bias=True,
                    ))
            else:
                attentions.append(None)

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                if input_ndim == 4:
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                hidden_states += attn(hidden_states, temb=temb)
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states