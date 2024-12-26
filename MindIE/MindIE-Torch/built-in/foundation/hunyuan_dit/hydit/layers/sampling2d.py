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

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample2D(nn.Module):

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        if norm_type == "rms_norm":
            raise ValueError(f"not support norm_type: {norm_type} yet.")
        if norm_type and norm_type != "ln_norm":
            raise ValueError(f"unknown norm_type: {norm_type}.")
        
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)

        elif norm_type is None:
            self.norm = None
        

        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            if self.channels != self.out_channels:
                raise ValueError("The parameter channels should be equal to out_channels")
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if hidden_states.shape[1] != self.channels:
            raise ValueError("The parameter hidden_states.shape[1] should equal to self.channels")
        
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        hidden_states = self.conv(hidden_states)

        return hidden_states


class Upsample2D(nn.Module):

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.interpolate = interpolate
        if norm_type == "rms_norm":
            raise ValueError(f"not support norm_type: {norm_type} yet.")
        if norm_type and norm_type != "ln_norm":
            raise ValueError(f"unknown norm_type: {norm_type}.")
        
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        if hidden_states.shape[1] != self.channels:
            raise ValueError("The parameter hidden_states.shape[1] should equal to self.channels")

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Do not need cast to float32 as 'upsample_nearest2d_out_frame' op support bfloat16 after PyTorch 2.1
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984 
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states