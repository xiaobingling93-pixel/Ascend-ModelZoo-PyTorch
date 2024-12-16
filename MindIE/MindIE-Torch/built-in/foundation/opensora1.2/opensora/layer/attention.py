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
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import logging
import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch_npu

from .norm import get_normalization_helper, LlamaRMSNorm
from .embdding import get_embedding_helper
from ..utils.utils import is_npu_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_TOKENS = 2147483647


class Attention(nn.Module):
    def __init__(
            self,
            dimension: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            norm_layer: nn.Module = LlamaRMSNorm,
            enable_flash_attn: bool = False,
            rope=None,
    ) -> None:
        super().__init__()
        if dimension % num_heads != 0:
            logger.error("dimension should be divisible by num_heads")
            raise ValueError('dimension should be divisible by num_heads')
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.scale = self.head_dim ** -0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dimension, dimension * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dimension, dimension)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def t_flash_attention(self, q, k, v):
        x = torch_npu.npu_fusion_attention(
            q, k, v, self.num_heads, input_layout="BNSD",
            pse=None,
            scale=self.scale,
            pre_tockens=MAX_TOKENS,
            next_tockens=MAX_TOKENS,
            keep_prob=1.,
            sync=False,
            inner_precise=0,
        )[0]

        x = x.transpose(1, 2)
        return x

    def s_flash_attention(self, q, k, v):
        x = torch_npu.npu_prompt_flash_attention(
            q, k, v, num_heads=self.num_heads,
            input_layout="BNSD",
            scale_value=1.0 / math.sqrt(self.head_dim),
            pre_tokens=MAX_TOKENS,
            next_tokens=MAX_TOKENS,
            sparse_mode=0)
        x = x.transpose(1, 2)
        return x

    def no_fused_flash_attention(self, q, k, v):
        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        x = attn @ v
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape0_b, x_shape1_n, x_shape2_c = x.shape
        enable_flash_attn = self.enable_flash_attn
        qkv = self.qkv(x)
        qkv_shape = (x_shape0_b, x_shape1_n, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        if enable_flash_attn:
            if is_npu_available() and q.dtype in [torch.float16, torch.bfloat16]:
                if self.rope:
                    x = self.t_flash_attention(q, k, v)
                else:
                    x = self.s_flash_attention(q, k, v)
            else:
                from flash_attn import flash_attn_func

                # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    softmax_scale=self.scale,
                )
        else:
            x = self.no_fused_flash_attention(q, k, v)

        x_output_shape = (x_shape0_b, x_shape1_n, x_shape2_c)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        if num_heads == 0:
            logger.error("num_heads cannot be zero")
            raise ValueError('num_heads cannot be zero')
        if d_model % num_heads != 0:
            logger.error("d_model must be divisible by num_heads")
            raise ValueError('d_model must be divisible by num_heads')
        if d_model // num_heads <= 0:
            logger.error("head_dim must be a positive integero")
            raise ValueError('head_dim must be a positive integero')

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        x_shape0_b, x_shape1_n, x_shape2_c = x.shape

        if is_npu_available() and x.dtype in [torch.float16, torch.bfloat16]:
            q = self.q_linear(x).view(-1, self.num_heads, self.head_dim)
            kv = self.kv_linear(cond).view(-1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(1)

            actual_seq_qlen = []
            actual_seq_kvlen = []
            if mask is not None:
                ans = 0
                for _ in range(x_shape0_b):
                    ans += x_shape1_n
                    actual_seq_qlen.append(ans)
                ans = 0
                for m in mask:
                    ans += m
                    actual_seq_kvlen.append(ans)

            x = torch_npu.npu_fusion_attention(
                q, k, v, self.num_heads, input_layout="TND",
                pse=None,
                scale=1.0 / math.sqrt(self.head_dim),
                pre_tockens=MAX_TOKENS,
                next_tockens=MAX_TOKENS,
                actual_seq_qlen=tuple(actual_seq_qlen),
                actual_seq_kvlen=tuple(actual_seq_kvlen),
                keep_prob=1.,
                sparse_mode=0,
            )[0]
        else:
            q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
            kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)

            attn_bias = None
            if mask is not None:
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([x_shape1_n] * x_shape0_b, mask)
            x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        x = x.view(x_shape0_b, -1, x_shape2_c)
        x = self.proj(x)
        return x



class AttnProcessor:
    """
    The standard attention processor.
    """
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states is None:
            logger.error("`hidden_states` can not be None.")
            raise ValueError("`hidden_states` can not be None.")

        # only support BNC now.
        if hidden_states.ndim != 3: # 3: BNC.
            logger.error("`hidden_states` dim must be 3, but got %d", hidden_states.ndim)
            raise ValueError(f"`hidden_states` dim must be 3, but got {hidden_states.ndim}")
        
        batch_size = hidden_states.shape[0]
        
        if attn.is_cross_attention:
            query = attn.q_proj(hidden_states)
            kv = attn.kv_proj(encoder_hidden_states)
            key, value = kv.reshape(batch_size, -1, 2, attn.num_heads, attn.head_dim).unbind(2) # B S 2 H
        else:
            qkv = attn.qkv_proj(hidden_states)
            query, key, value = qkv.reshape(batch_size, -1, 3, attn.num_heads, attn.head_dim).unbind(2) # B S 3 H
        query = query.reshape(batch_size, -1, attn.num_heads, attn.head_dim).transpose(1, 2) # BNSD
        key = key.reshape(batch_size, -1, attn.num_heads, attn.head_dim).transpose(1, 2) # BNSD
        value = value.reshape(batch_size, -1, attn.num_heads, attn.head_dim).transpose(1, 2) # BNSD
        
        # norm q and k
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # position embedding q and k
        query = attn.position_embedding(query)
        key = attn.position_embedding(key)

        # need replaced by dispatch flash_attention function
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=attn.scale_value)
        # transform the hidden_states layout from BNSD to BSH
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.num_heads * attn.head_dim)
        hidden_states = attn.out_proj(hidden_states)
        return hidden_states
