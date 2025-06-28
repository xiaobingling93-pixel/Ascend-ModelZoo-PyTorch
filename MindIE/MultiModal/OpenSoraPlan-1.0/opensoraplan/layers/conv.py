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
import math
from typing import Union, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from opensoraplan.utils.log import logger
from .ops import cast_tuple
from .ops import video_to_image


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        
    @video_to_image
    def forward(self, x):
        return super().forward(x)
        

class CausalConv3d(nn.Module):
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], init_method="random", **kwargs
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        stride = kwargs.pop("stride", 1)
        padding = kwargs.pop("padding", 0)
        padding = list(cast_tuple(padding, 3))
        padding[0] = 0
        stride = cast_tuple(stride, 3)
        self.conv = nn.Conv3d(chan_in, chan_out, self.kernel_size, stride=stride, padding=padding)
        self._init_weights(init_method)
        self.embed_dim = self.chan_out
        self.patch_size = self.kernel_size
        self.stride = stride
        self.padding = padding 
            
    def forward(self, x):
        # 1 + 16   16 as video, 1 as image
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )   # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)

        def generate_random_conv3d_output(input_shape, out_channels, kernel_size, stride, padding):
            n, _, d, h, w = input_shape
            k_d, k_h, k_w = kernel_size
            s_d, s_h, s_w = stride
            p_d, p_h, p_w = padding

            d_out = math.floor((d + 2 * p_d - k_d) / s_d + 1)
            h_out = math.floor((h + 2 * p_h - k_h) / s_h + 1)
            w_out = math.floor((w + 2 * p_w - k_w) / s_w + 1)

            output_shape = (n, out_channels, d_out, h_out, w_out)
            return torch.rand(output_shape, dtype=x.dtype, device=x.device)
        
      
        return self.conv(x)
    
    def _init_weights(self, init_method):
        ks = torch.tensor(self.kernel_size)
        if init_method == "avg":
            if not (self.kernel_size[1] == 1 and self.kernel_size[2] == 1):
                logger.error("only support temporal up/down sample")
                raise ValueError
            if self.chan_in != self.chan_out:
                logger.error("chan_in must be equal to chan_out")
                raise ValueError
            weight = torch.zeros((self.chan_out, self.chan_in, *self.kernel_size))

            eyes = torch.concat(
                [
                    torch.eye(self.chan_in).unsqueeze(-1) * 1 / 3,
                    torch.eye(self.chan_in).unsqueeze(-1) * 1 / 3,
                    torch.eye(self.chan_in).unsqueeze(-1) * 1 / 3,
                ],
                dim=-1,
            )
            weight[:, :, :, 0, 0] = eyes

            self.conv.weight = nn.Parameter(
                weight,
                requires_grad=True,
            )
        elif init_method == "zero":
            self.conv.weight = nn.Parameter(
                torch.zeros((self.chan_out, self.chan_in, *self.kernel_size)),
                requires_grad=True,
            )
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)