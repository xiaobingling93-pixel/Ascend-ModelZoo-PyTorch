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


class QKVLinear(nn.Module):
    def __init__(self, attention_dim, hidden_size, qkv_bias=True, device=None, dtype=None):
        super(QKVLinear, self).__init__()
        self.attention_dim = attention_dim
        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty([self.attention_dim, 3 * self.hidden_size], **factory_kwargs))
        if self.qkv_bias:
            self.bias = nn.Parameter(torch.empty([3 * self.hidden_size], **factory_kwargs))

    def forward(self, hidden_states):

        if not self.qkv_bias:
            qkv = torch.matmul(hidden_states, self.weight)
        else:
            qkv = torch.addmm(
                self.bias, 
                hidden_states.view(hidden_states.size(0) * hidden_states.size(1), hidden_states.size(2)),
                self.weight, 
                beta=1, 
                alpha=1
            )

        return qkv