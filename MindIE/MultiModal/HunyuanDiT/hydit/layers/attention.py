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

from mindiesd import rotary_position_embedding, attention_forward
from .norm import get_normalization_helper

EPS_DEFAULT = 1e-6
EPS_FP16 = 1 / 65530


class Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 cross_attention_dim: int,
                 num_heads: int = 16,
                 attention_norm: str = "layer_norm",
                 rotated_mode: str = "rotated_half",
                 qkv_bias: bool = True): 
        super().__init__()

        self.is_cross_attention = cross_attention_dim is not None
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        if not self.is_cross_attention:
            self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=qkv_bias)
        else:
            self.kv_proj = nn.Linear(cross_attention_dim, 2 * hidden_size, bias=qkv_bias)

        # If using fp16, eps should be 1 / 65530; else default 1e-6
        self.q_norm = get_normalization_helper(attention_norm, self.head_dim, eps=EPS_FP16)
        self.k_norm = get_normalization_helper(attention_norm, self.head_dim, eps=EPS_FP16)

        self.rotated_mode = rotated_mode

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)


    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor = None,
                freqs_cis_img: torch.Tensor = None,
                layer: int = 0):
        # hidden_states, encoder_hidden_states dtype: float16
        if hidden_states is None:
            raise ValueError("Input hidden_states should not be none.")
        if freqs_cis_img is None:
            raise ValueError("Input freqs_cis_img should not be none.")
        cos, sin = freqs_cis_img

        # only support BNC now.
        if hidden_states.ndim != 3: # 3: BNC
            raise ValueError(f"The dimensions of hidden_states should be 3, but got {hidden_states.ndim}")

        batch_size = hidden_states.shape[0]

        query = self.q_proj(hidden_states)
        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim)
        if not self.is_cross_attention:
            kv = self.kv_proj(hidden_states)
        else:
            kv = self.kv_proj(encoder_hidden_states)
        key, value = kv.reshape(batch_size, -1, 2, self.num_heads, self.head_dim).unbind(2)
        # query, key, value dtype: float16

        query = self.q_norm(query)
        key = self.k_norm(key)

        # position embedding q and k, and flash attention
        query = rotary_position_embedding(query, cos, sin, rotated_mode=self.rotated_mode, head_first=False)
        if not self.is_cross_attention:
            key = rotary_position_embedding(key, cos, sin, rotated_mode=self.rotated_mode, head_first=False)
            hidden_states = attention_forward(query, key, value, opt_mode="manual",
                                              op_type="fused_attn_score", layout="BNSD")
        else:
            hidden_states = attention_forward(query, key, value, opt_mode="manual",
                                              op_type="fused_attn_score", layout="BSND")
        hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.out_proj(hidden_states)
        return hidden_states
