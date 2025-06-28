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
from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .ops import cast_tuple
from .conv import CausalConv3d


class SpatialDownsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (2, 2),
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        stride = cast_tuple(stride, 2)
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


def mock_interpolate(input_tensor, scale_factor, mode="nearest"):
    """
    仿真 interpolate函数,返回一个具有正确shape 的随机数张量。

    :param input_tensor: 输入张量（可以使三维或者四维）
    :param scale_factor: 缩放因子，元组形式 (height_scale, width_scale) 或者单个数值
    :param mode: 插值模式(仅用于兼容性， 不影响仿真的输出)
    :return: 具有预期形状的随机数张量
    """
    # 获取输入张量的形状
    input_shape = input_tensor.shape 
    ndim = len(input_shape)

    if ndim == 4:
        # 四维张量(batch_size, channels, height, width)
        batch_size, channels, height, width = input_shape
        output_height = int(height * scale_factor[0])
        output_width = int(width * scale_factor[1])
        output_shape = (batch_size, channels, output_height, output_width)
    elif ndim == 3:
        # 三维张量(batch_size, length, feature_dim)
        batch_size, length, feature_dim = input_shape
        if isinstance(scale_factor, tuple):
            output_length = int(length * scale_factor[0])
        else:
            output_length = int(length * scale_factor)
        output_shape = (batch_size, output_length, feature_dim)
    else:
        raise ValueError("仅支持三维或四维张量")

    # 创建一个具有预期形状的随机数张量
    output_tensor = torch.randn(output_shape)

    return output_tensor


class SpatialUpsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (1, 1),
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=1
        )

    def forward(self, x):
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> b (c t) h w")
        x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
        x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        x = self.conv(x)
        return x


class TimeDownsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int = 3
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.AvgPool3d((kernel_size, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.kernel_size - 1, 1, 1)
        )
        x = torch.concatenate((first_frame_pad, x), dim=2)
        return self.conv(x)


class TimeUpsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out
    ):
        super().__init__()

    def forward(self, x):
        if x.size(2) > 1:
            x, x_ = x[:, :, :1], x[:, :, 1:]
            x_ = F.interpolate(x_, scale_factor=(2, 1, 1), mode='trilinear')
            x = torch.concat([x, x_], dim=2)
        return x


class TimeDownsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 2,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.avg_pool = nn.AvgPool3d((kernel_size, 1, 1), stride=(2, 1, 1))
        self.conv = nn.Conv3d(
            in_channels, out_channels, self.kernel_size, stride=(2, 1, 1), padding=(0, 1, 1)
        )
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.kernel_size[0] - 1, 1, 1)
        )
        x = torch.concatenate((first_frame_pad, x), dim=2)
        return alpha * self.avg_pool(x) + (1 - alpha) * self.conv(x)


class TimeUpsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 2,
    ):
        super().__init__()
        self.conv = CausalConv3d(
            in_channels, out_channels, kernel_size, padding=1
        )
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        if x.size(2) > 1:
            x, x_ = x[:, :, :1], x[:, :, 1:]
            x_ = F.interpolate(x_, scale_factor=(2, 1, 1), mode='trilinear')
            x = torch.concat([x, x_], dim=2)
        return alpha * x + (1 - alpha) * self.conv(x)
