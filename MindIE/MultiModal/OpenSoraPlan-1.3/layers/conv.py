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

import math
from typing import Optional, Tuple, Union
from collections import deque

import torch
import torch.nn as nn
import torch_npu
from .utils import video_to_image, cast_tuple
from .norm import Normalize

    
class VideoConv2d(nn.Conv2d):
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


class PlanCausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=False,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3))
        self.padding[0] = 0
        self.stride = cast_tuple(self.stride, 3)
        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias
        )
        self.enable_cached = enable_cached
        self.is_first_chunk = True
        
        self.causal_cached = deque()
        self.cache_offset = 0

    def forward(self, x):
        if self.is_first_chunk:
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, self.time_kernel_size - 1, 1, 1)
            )
        else:
            first_frame_pad = self.causal_cached.popleft()
        x = torch.concatenate((first_frame_pad, x), dim=2)

        if self.enable_cached and self.time_kernel_size != 1:
            if (self.time_kernel_size - 1) // self.stride[0] != 0:
                if self.cache_offset == 0:
                    self.causal_cached.append(x[:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
                else:
                    self.causal_cached.append(x[
                        :, :, :-self.cache_offset][:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
            else:
                self.causal_cached.append(x[:, :, 0:0, :, :].clone())
        elif self.enable_cached:
            self.causal_cached.append(x[:, :, 0:0, :, :].clone())

        x = self.conv(x)
        return x
    

class AttnBlock3DFix(nn.Module):

    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = PlanCausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = PlanCausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = PlanCausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = PlanCausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        
        attn_output = torch_npu.npu_fused_infer_attention_score(q, k, v,
            atten_mask=None, input_layout="BSH", scale=1 / math.sqrt(c),
            num_heads=1)[0]

        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        h_ = self.proj_out(attn_output)

        return x + h_