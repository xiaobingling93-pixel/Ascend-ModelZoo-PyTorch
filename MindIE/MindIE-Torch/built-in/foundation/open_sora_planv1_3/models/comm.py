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

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist

from .parallel_mgr import get_sequence_parallel_size, get_sequence_parallel_group


def _all_to_all_func(input_, world_size, process_group, scatter_dim=2, gather_dim=1):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=process_group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def split_sequence(input_, process_group: dist.ProcessGroup, dim: int, pad: int):
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    if world_size == 1:
        return input_
    
    if pad > 0:
        pad_size = list(input_.shape)
        pad_size[dim] = pad
        input_ = torch.cat([input_, torch.zeros(pad_size, dtype=input_.dtype, device=input_.device)], dim=dim)
    
    dim_size = input_.size(dim)
    if dim_size % world_size != 0:
        raise ValueError(
            f"The th{dim} dimensions of input_:{input_.size()} is not divisible by world_size:{world_size}.")

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()
    return output


def gather_sequence(input_, process_group: dist.ProcessGroup, dim: int, pad: int):
    input_ = input_.contiguous()
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_
    
    #all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=process_group)

    #concat
    output = torch.cat(tensor_list, dim=dim)

    if pad > 0:
        output = output.narrow(dim, 0, output.size(dim) - pad)
    
    return output

# ======
# Pad
# ======

SPTIAL_PAD = 0
TEMPORAL_PAD = 0


def set_spatial_pad(dim_size: int):
    sp_size = get_sequence_parallel_size()
    pad = (sp_size - (dim_size % sp_size)) % sp_size
    global SPTIAL_PAD
    SPTIAL_PAD = pad


def get_spatial_pad() -> int:
    return SPTIAL_PAD


def set_temporal_pad(dim_size: int):
    sp_size = get_sequence_parallel_size()
    pad = (sp_size - (dim_size % sp_size)) % sp_size
    global TEMPORAL_PAD
    TEMPORAL_PAD = pad


def get_temporal_pad() -> int:
    return TEMPORAL_PAD


def all_to_all_with_pad(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    **kwargs
):  
    scatter_dim = kwargs.get("scatter_dim", 2)
    gather_dim = kwargs.get("gather_dim", 1)
    scatter_pad = kwargs.get("scatter_pad", 0)
    gather_pad = kwargs.get("gather_pad", 0)

    if scatter_pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[scatter_dim] = scatter_pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        input_ = torch.cat([input_, pad_tensor], dim=scatter_dim)

    world_size = dist.get_world_size(process_group)
    if input_.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"The scatter_dim:{scatter_dim} of input_:{input_.shape} is not divisible by world_size:{world_size}.")

    input_ = _all_to_all_func(input_, world_size, process_group, scatter_dim, gather_dim)

    if gather_pad > 0:
        input_ = input_.narrow(gather_dim, 0, input_.size(gather_dim) - gather_pad)
    
    return input_


def all_to_all(
    tensor: torch.Tensor,
    world_size: int,
    scatter_dim: int,
    gather_dim: int,
    process_group: dist.ProcessGroup = None,
):
    if process_group is None:
        process_group = dist.group.WORLD
    return _all_to_all_func(tensor, world_size, process_group, scatter_dim, gather_dim)


def all_to_all_sbh(
    input_: torch.Tensor,
    scatter_dim: int = 1,
    gather_dim: int = 0,
):
    return single_all_to_all(input_, scatter_dim, gather_dim)


def single_all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
):

    sp_size = get_sequence_parallel_size()
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // sp_size
    if scatter_dim < 1:
        input_t = input_.reshape(
            [sp_size, inp_shape[scatter_dim]] + \
            inp_shape[scatter_dim + 1:]
        )
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input_.reshape(
            [-1, sp_size, inp_shape[scatter_dim]] + \
            inp_shape[scatter_dim + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)

    dist.all_to_all_single(output, input_t, group=get_sequence_parallel_group())
    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_dim < 1:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_dim] + [inp_shape[gather_dim] * sp_size, ] + inp_shape[gather_dim + 1:])