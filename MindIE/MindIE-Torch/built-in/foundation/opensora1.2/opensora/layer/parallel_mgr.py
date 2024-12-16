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

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup

PARALLEL_MANAGER = None


class ParallelManager(ProcessGroupMesh):
    def __init__(self, sp_size, sp_axis):
        super().__init__(sp_size)
        self.sp_size = sp_size
        self.sp_axis = sp_axis
        self.sp_group: ProcessGroup = self.get_group_along_axis(sp_axis)
        self.sp_rank = dist.get_rank(self.sp_group)
        self.enable_sp = sp_size > 1


def set_parallel_manager(sp_size, sp_axis):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(sp_size, sp_axis)


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_sequence_parallel_size():
    return PARALLEL_MANAGER.sp_size


def get_sequence_parallel_rank():
    return PARALLEL_MANAGER.sp_rank


def use_sequence_parallel():
    return PARALLEL_MANAGER.enable_sp


def get_parallel_manager():
    return PARALLEL_MANAGER