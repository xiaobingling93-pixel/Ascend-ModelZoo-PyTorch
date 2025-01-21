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

import os 

import torch_npu
import torch.distributed as dist


class ParallelManager():
    def __init__(self, world_size=1, rank=0, group=None):
        self.sp_size = world_size
        self.sp_group = group
        self.enable_sp = world_size > 1
        self.rank = rank


PARALLEL_MANAGER = ParallelManager()


def set_parallel_manager(world_size, rank, group):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(world_size, rank, group)


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_sequence_parallel_size():
    return PARALLEL_MANAGER.sp_size


def get_sequence_parallel_state():
    return PARALLEL_MANAGER.enable_sp


def get_sequence_parallel_rank():
    return PARALLEL_MANAGER.rank


def init_parallel_env(enable_sequence_parallelism):
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend='hccl', init_method='env://', 
        world_size=world_size, rank=rank
        )
    if enable_sequence_parallelism:
        set_parallel_manager(world_size, rank, dist.group.WORLD)