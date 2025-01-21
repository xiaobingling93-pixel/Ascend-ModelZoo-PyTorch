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
import torch_npu


class QKVLinear(nn.Module):
    def __init__(self, attention_dim, hidden_size, qkv_bias=True, cross_attention_dim=None, device=None, dtype=None):
        super(QKVLinear, self).__init__()
        self.attention_dim = attention_dim
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.qkv_bias = qkv_bias

        factory_kwargs = {"device": device, "dtype": dtype}

        if not cross_attention_dim:
            self.weight = nn.Parameter(torch.empty([self.attention_dim, 3 * self.hidden_size], **factory_kwargs))
            if self.qkv_bias:
                self.bias = nn.Parameter(torch.empty([3 * self.hidden_size], **factory_kwargs))
        else:
            self.q_weight = nn.Parameter(torch.empty([self.attention_dim, self.hidden_size], **factory_kwargs))
            self.kv_weight = nn.Parameter(torch.empty([self.attention_dim, 2 * self.hidden_size], **factory_kwargs))

            if self.qkv_bias:
                self.q_bias = nn.Parameter(torch.empty([self.hidden_size], **factory_kwargs))
                self.kv_bias = nn.Parameter(torch.empty([2 * self.hidden_size], **factory_kwargs))


    def forward(self, hidden_states, encoder_hidden_states=None):

        if self.cross_attention_dim is None:
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

            batch, seqlen, _ = hidden_states.shape
            qkv_shape = (batch, seqlen, 3, -1)
            qkv = qkv.view(qkv_shape)
            q, k, v = qkv.unbind(2)

        else:
            if not self.qkv_bias:
                q = torch.matmul(hidden_states, self.q_weight)
                kv = torch.matmul(encoder_hidden_states, self.kv_weight)
            else:
                q = torch.addmm(
                    self.q_bias, 
                    hidden_states.view(hidden_states.size(0) * hidden_states.size(1), hidden_states.size(2)), 
                    self.q_weight, 
                    beta=1, 
                    alpha=1
                )
                kv = torch.addmm(
                    self.kv_bias, 
                    encoder_hidden_states.view(
                        encoder_hidden_states.size(0) * encoder_hidden_states.size(1),
                        encoder_hidden_states.size(2)), 
                    self.kv_weight, 
                    beta=1, 
                    alpha=1
                )

            batch, seqlen, _ = encoder_hidden_states.shape
            kv_shape = (batch, seqlen, 2, -1)

            kv = kv.view(kv_shape)
            k, v = kv.unbind(2)

            batch, seqlen, _ = hidden_states.shape
            q = q.view(batch, seqlen, -1)

        return q, k, v