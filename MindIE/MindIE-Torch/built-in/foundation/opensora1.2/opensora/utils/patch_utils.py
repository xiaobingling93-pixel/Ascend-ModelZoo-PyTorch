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

import torch
import torch.nn as nn
import torch.distributed as dist


class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, hidden_state, dim, is_overlap):
        length = hidden_state.shape[dim]
        if is_overlap:
            overlap = self.rank % 2
            start_idx = (length + self.world_size - 1) // self.world_size * self.rank - overlap
            end_idx = min((length + self.world_size - 1) // self. world_size * (self.rank + 1) - overlap + 1, length)
        else:
            start_idx = (length + self.world_size - 1) // self.world_size * self.rank
            end_idx = min((length + self.world_size - 1) // self.world_size * (self.rank + 1), length)
        idx = torch.arange(start_idx, end_idx, device=f"npu:{self.rank}")
        return hidden_state.index_select(dim, idx).clone()
        

class Depatchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def forward(self, patch_hidden_state, dim, is_overlap):
        if is_overlap:
            overlap = self.rank % 2
            start_idx = overlap
            end_idx = patch_hidden_state.shape[dim] + overlap - 1
            idx = torch.arange(start_idx, end_idx, device=f"npu:{self.rank}")
            patch_hidden_state = patch_hidden_state.index_select(dim, idx)
        
        patch_length_list = [torch.empty([1], dtype=torch.int64, device=f"npu:{self.rank}") 
                             for _ in range(self.world_size)]
        dist.all_gather(
            patch_length_list,
            torch.tensor(
                [patch_hidden_state.shape[dim]],
                dtype=torch.int64,
                device=f"npu:{self.rank}"
            )
        )
        patch_shape = list(patch_hidden_state.shape)
        patch_hidden_state_list = []
        for i in range(self.world_size):
            patch_shape[dim] = patch_length_list[i].item()
            patch_hidden_state_list.append(
                torch.empty(tuple(patch_shape), dtype=patch_hidden_state.dtype, device=f"npu:{self.rank}"))
        dist.all_gather(
            patch_hidden_state_list,
            patch_hidden_state.contiguous()
        )

        return torch.cat(patch_hidden_state_list, dim)