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

from typing import Union, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from .utils import video_to_image
from .conv import PlanCausalConv3d


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.with_conv = True
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
            
    @video_to_image
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, undown=False):
        super().__init__()
        self.with_conv = True
        self.undown = undown
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            if self.undown:
                self.conv = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
            else:
                self.conv = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=0)

    @video_to_image
    def forward(self, x):
        if self.with_conv:
            if self.undown:
                x = self.conv(x)
            else:
                pad = (0, 1, 0, 1)
                x = F.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Spatial2xTime2x3DDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = PlanCausalConv3d(in_channels, out_channels, kernel_size=3, padding=0, stride=2)

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Spatial2xTime2x3DUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_interpolation="trilinear",
        enable_cached=False,
    ):
        super().__init__()
        self.t_interpolation = t_interpolation
        self.conv = PlanCausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.enable_cached = enable_cached
        self.causal_cached = deque()

    def forward(self, x):
        mode_method = "trilinear"
        if x.size(2) > 1 or self.causal_cached is not None :
            if self.enable_cached and len(self.causal_cached) > 0:
                x = torch.cat([self.causal_cached.popleft(), x], dim=2)
                self.causal_cached.append(x[:, :, -2:-1].clone())
                x = F.interpolate(x, scale_factor=(2, 1, 1), mode=self.t_interpolation)
                x = x[:, :, 2:]
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode=mode_method)
            else:
                if self.enable_cached:
                    self.causal_cached.append(x[:, :, -1:].clone())
                x, x_ = x[:, :, :1], x[:, :, 1:]
                x_ = F.interpolate(
                    x_, scale_factor=(2, 1, 1), mode=self.t_interpolation
                )
                x_ = F.interpolate(x_, scale_factor=(1, 2, 2), mode=mode_method)
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode=mode_method)
                x = torch.concat([x, x_], dim=2)
        else:
            if self.enable_cached:
                self.causal_cached.append(x[:, :, -1:].clone())
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode=mode_method)
        return self.conv(x)