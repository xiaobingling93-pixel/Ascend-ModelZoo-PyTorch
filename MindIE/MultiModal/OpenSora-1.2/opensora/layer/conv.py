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

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.utils import _triple, _reverse_repeat_tuple
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_3_t


class Conv3dAdapter(nn.Module):
    def __init__(
        self,
        conv3d: nn.Conv3d,
        is_casual=False,
        block_size=2,
    ):
        super().__init__()
        self.module = PatchConv3d(
            in_channels=conv3d.in_channels,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=conv3d.padding,
            dilation=conv3d.dilation,
            groups=conv3d.groups,
            bias=conv3d.bias is not None,
            padding_mode=conv3d.padding_mode,
            device=conv3d.weight.device,
            dtype=conv3d.weight.dtype,
            block_size=block_size,
            is_casual=is_casual,
        )
        self.module.weight.data = conv3d.weight.data
        if conv3d.bias is not None:
            self.module.bias.data = conv3d.bias.data

    def forward(self, x):
        return self.module(x)


class PatchConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int]] = 2,
        is_casual: bool = False,
        is_overlap: bool = True
    ) -> None:
        self.padding = padding if isinstance(padding, str) else _triple(padding)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.block_size = block_size
        self.is_casual = is_casual
        self.is_overlap = is_overlap
        self.rank = 0
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, self.kernel_size,
                                   range(len(self.kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        # initialize weight and bias
        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *self.kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *self.kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ch_in, ch_out, *_ = self.weight.shape
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = math.prod([item for item in self.kernel_size]) * ch_out
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        
    def forward(self, patch_hidden_state: Tensor, weight: Tensor = None, bias: Tensor = None) -> Tensor:
        if weight is None:
            return self._conv_forward(patch_hidden_state, self.weight, self.bias)
        else:
            return self._conv_forward(patch_hidden_state, weight, bias)
    
    def _one_worldsize_conv(self, padding_mode, patch_hidden_state, weight, bias):
        if padding_mode != 'zeros':
            return F.conv3d(F.pad(patch_hidden_state, self._reversed_padding_repeated_twice, 
                                    mode=padding_mode), weight, bias, self.stride, 
                                    _triple(0), self.dilation, self.groups)
        return F.conv3d(patch_hidden_state, weight, bias, self.stride, 
                        self.padding, self.dilation, self.groups)

    def _pre_conv_forward(self, patch_hidden_state, shape):
        bs, channels, t, h, _ = shape
        if self.rank % 2 == 0 and self.rank != 0:
            send = patch_hidden_state[..., :1].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank - 1)
            recv = torch.zeros([bs, channels, t, h, 1], 
                dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank - 1)
            dist.batch_isend_irecv([send_op, recv_op])
            return recv
        elif self.rank % 2 != 0 and self.rank != self.world_size - 1:
            send = patch_hidden_state[..., -1:].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank + 1)
            recv = torch.zeros([bs, channels, t, h, 1], 
                dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank + 1)
            dist.batch_isend_irecv([send_op, recv_op])
            return recv
        return None
        

    def _end_conv_forward(self, outputs, shape):  
        bs_, channels_, t_, h_, _ = shape
        if self.rank % 2 == 0:
            send = outputs[0][..., -1:].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank + 1)
            recv = torch.zeros([bs_, channels_, t_, h_, 1], 
                dtype=outputs[0].dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank + 1)
            dist.batch_isend_irecv([send_op, recv_op])
        else:
            send = outputs[0][..., :1].contiguous()
            send_op = dist.P2POp(dist.isend, send, self.rank - 1)
            recv = torch.zeros([bs_, channels_, t_, h_, 1], 
                dtype=outputs[0].dtype, device=f"npu:{self.rank}")
            recv_op = dist.P2POp(dist.irecv, recv, self.rank - 1)
            dist.batch_isend_irecv([send_op, recv_op])
        return recv

    def _parallel_conv_forward(self, patch_hidden_state, weight, bias):
        shape = patch_hidden_state.shape
        bs, channels, t, h, w = shape
        patch_hidden_state, padding = self._adjust_padding_for_patch(patch_hidden_state, self.padding)
        stride = (w - 1 + self.block_size - 1) // self.block_size
        overlap = self.kernel_size[0] // 2
        outputs = []
        recv = None
        # P2P communication
        for step in range(self.block_size):
            start_idx = step * stride + 1 - overlap
            end_idx = min((step + 1) * stride + 1 + overlap, w)
            if self.rank % 2 == 0:
                input_patch = patch_hidden_state[..., w - end_idx:w - start_idx]
            else:
                input_patch = patch_hidden_state[..., start_idx:end_idx]

            if step == 0:
                recv = self._pre_conv_forward(patch_hidden_state, shape)
            if step == self.block_size - 1:
                if overlap == 1:
                    input_patch = torch.cat([recv, input_patch], dim=-1) \
                        if self.rank % 2 == 0 else torch.cat([input_patch, recv], dim=-1)
                recv = self._end_conv_forward(outputs, outputs[0].shape)
            
            outputs.append(F.conv3d(input_patch, weight, bias, self.stride, padding, self.dilation, self.groups))

            if step == 0:
                if self.rank == 0:
                    recv = torch.zeros([bs, channels, t, h, 1],
                        dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
                elif self.rank == self.world_size - 1:
                    recv = torch.zeros([bs, channels, t, h, 1],
                        dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}")
            if step == self.block_size - 1:
                if self.rank % 2 == 0:
                    outputs.insert(0, recv)
                    outputs.reverse()
                else:
                    outputs.insert(0, recv)

        return torch.cat(outputs, dim=-1)

    def _conv_forward(self, patch_hidden_state: Tensor, weight: Tensor, bias: Optional[Tensor]):
        self._get_world_size_and_rank()
        if (self.world_size == 1):
            return self._one_worldsize_conv(self.padding_mode, patch_hidden_state, weight, bias)
        else:
            return self._parallel_conv_forward(patch_hidden_state, weight, bias)
            
    def _get_world_size_and_rank(self):
        world_size = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.world_size = world_size
        self.rank = rank
    
    def _adjust_padding_for_patch(self, patch_input, padding):
        if self.kernel_size[-1] == 3 and self.is_casual:
            patch_input = patch_input[..., 1:-1]
        padding = list(padding)
        padding[-1] = 0
        return patch_input, tuple(padding)