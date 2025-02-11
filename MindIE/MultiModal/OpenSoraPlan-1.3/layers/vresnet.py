#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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

import torch
import torch.nn as nn

from .norm import Normalize
from .conv import PlanCausalConv3d
from .utils import video_to_image

from .activation import get_activation_fn


class VideoResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        norm_type,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = get_activation_fn("silu")

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    @video_to_image
    def forward(self, x):
        h = x
        #CAST ?
        h = self.norm1(h)

        h = self.nonlinearity(h)
        h = self.conv1(h)
        #CAST ? 
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        x = x + h
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        norm_type,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = PlanCausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.conv2 = PlanCausalConv3d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = get_activation_fn("silu")
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = PlanCausalConv3d(
                    in_channels, out_channels, 3, padding=1
                )
            else:
                self.nin_shortcut = PlanCausalConv3d(
                    in_channels, out_channels, 1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        #CAST float32 ? 
        h = self.norm2(h)

        h = self.nonlinearity(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h