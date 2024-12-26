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
import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch_npu

from .embedding import get_embedding_helper
from .linear import QKVLinear
from .norm import get_normalization_helper, LlamaRMSNorm
from ..utils import is_npu_available

ALIGNMENT_BASE = 16
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
            raise ValueError(f"Input dimension:{dimension} should be divisible by num_heads:{num_heads}.")

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
            inner_precise=0
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
                raise ValueError(
                    f"Unsupported configuration: NPU available: {is_npu_available()}.")
        else:
            x = self.no_fused_flash_attention(q, k, v)

        x_output_shape = (x_shape0_b, x_shape1_n, x_shape2_c)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        return x


class ReconstitutionAttention(nn.Module):
    r"""
    Attention layer.
    """
    def __init__(
        self,
        attention_dim: int,
        cross_attention_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = True,
        out_proj_bias: bool = True,
        num_norm_groups: Optional[int] = None,
        attention_norm: Optional[str] = None,
        position_embedding: Optional[str] = None,
        add_proj_dim: Optional[int] = None,
        add_proj_bias: bool = True,
        enable_add_out_proj: bool = True,
        scale_attention: bool = True,
        eps: float = 1e-5,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
    ):
        r"""
        Attention layer init function.
        Args:
            attention_dim (`int`): 
                The number of channels in the hidden_states.
            cross_attention_dim (`int`, *optional*, defaults to `None`):
                The number of channels in the encoder_hidden_states. If not `None`, means cross attention.
            num_heads (`int`, *optional*, defaults to 8):
                The number of attention heads used in the multi-head attention layers.
            head_dim (`int`, *optional*, defaults to 64):
                The number of dims in each head.
            qkv_bias (`bool`, *optional*, defaults to `True`):
                Whether or not the quert, key and value linear layer to contain a bias parameter.
            out_proj_bias (`bool`, *optional*, defaults to `True`):
                Whether or not the out projection layer to contain a bias parameter.
            num_norm_groups (`int`, *optional*, defaults to `None`):
                The number of groups to use for the `GroupNorm` in attention.
                If `None`, no `GroupNorm` is used.
            attention_norm (`str`, *optional*, defaults to `None`):
                The type of normalization to use for the query and key in attention.
                Can be `None`, `layer_norm`, or `llama_rms_norm`.
            position_embedding (`str`, *optional*, defaults to `None`):
                The type of position embdding to use for the query and key in attention. Can be `None`, `rope`.
            add_proj_dim (`int`, *optional*, defaults to `None`):
                The number of channels to use for the additional projections. If `None`, no projection is used.
            add_proj_bias (`bool`, *optional*, defaults to `True`):
                Whether or not the additional projection layer to contain a bias parameter.
            enable_add_out_proj (`bool`, *optional*, defaults to `True`):
                Whether or not use the additional out projection.
            scale_attention (`bool`, *optional*, defaults to `True`):
                Set `True` to scale the query @ key result with by `1 / sqrt(head_dim)`.
            eps (`float`, *optional*, defaults to 1e-5):
                The additional value added to eh denominator in normalization.
            processor (`AttnProcessor`, *optional*, defaults to `None`):
                The attention processor to use. If `None`, `AttnProcessor` will be used.
        """
        super().__init__()

        self.num_heads = num_heads
        if head_dim <= 0:
            raise ValueError(f"Input head_dim should be greater than zero, but got {head_dim}.")
        self.head_dim = head_dim
        self.pad_dim = self.head_dim

        self.is_cross_attention = cross_attention_dim is not None

        self.scale_value = 1 / math.sqrt(head_dim) if scale_attention else 1.0

        # `hidden_size` is calculated by num_heads * head_dim -> H = N * D
        hidden_size = num_heads * head_dim

        hidden_size = self._set_pad(hidden_size, num_norm_groups, attention_norm, position_embedding)

        # Normalize hidden states by group_norm
        self.group_norm = nn.GroupNorm(num_channels=hidden_size, num_groups=num_norm_groups, eps=eps, affine=True) \
            if num_norm_groups is not None else None

        # Init normalization layer by get_normalization_helper.
        self.norm_q = get_normalization_helper(attention_norm, head_dim, eps)
        self.norm_k = get_normalization_helper(attention_norm, head_dim, eps)

        # Init position embedding by get_embedding_helper.
        self.position_embedding = get_embedding_helper(position_embedding, head_dim)

        # QKVLinear
        if self.is_cross_attention:
            self.qkv_proj = QKVLinear(attention_dim, hidden_size, qkv_bias, cross_attention_dim)
        else:
            self.qkv_proj = QKVLinear(attention_dim, hidden_size, qkv_bias)
        
        # Additional qkv linear for Multi-Modal Diffusion Transformer
        if add_proj_dim is not None:
            self.add_qkv_proj = QKVLinear(add_proj_dim, hidden_size, add_proj_bias)

        # OutLinear
        self.out_dim = out_dim if out_dim is not None else attention_dim
        self.out_proj = nn.Linear(hidden_size, self.out_dim, bias=out_proj_bias)
        
        # Additional out linear for Multi-Modal Diffusion Transformer
        if add_proj_dim is not None:
            # For the last attention layer in Multi-Modal Diffusion Transformer,
            # no need to calculate the additional out linear
            self.add_out_proj = nn.Linear(hidden_size, attention_dim, bias=out_proj_bias) \
                if enable_add_out_proj else nn.Identity()

        # Set default processor by AttnProcessor
        attn_processor = processor if processor is not None else AttnProcessor()
        self.set_processor(attn_processor)

    def set_processor(self, processor: "AttnProcessor"):
        """
        Set the attention processor.
        Users can develop different attention processor for `Attention` to achieve different functions.
        Args:
            processor: ("AttnProcessor"):
                The attention processor to used for attention forward.
        """
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Attention forward function.
        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of attention query.
            encoder_hidden_states (`torch.Tensor`, *optional*, defaults to `None`):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                The mask of attention.
            **kwargs:
                The additional arguments to the attention processors.
                For standard attention use `AttnProcessor`, kwargs is empty.
        Returns:
            `torch.Tensor`: The output of the attention layer. 
        """
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        attn_kwargs = {key: value for key, value in kwargs.items() if key in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **attn_kwargs
        )

    def _set_pad(self, hidden_size, num_norm_groups, attention_norm, position_embedding):
        if self.head_dim % ALIGNMENT_BASE == 0:
            return hidden_size
        elif (num_norm_groups is not None) or (attention_norm is not None) or (position_embedding is not None):
            return hidden_size
        else:
            self.pad_dim = (self.head_dim // ALIGNMENT_BASE + 1) * ALIGNMENT_BASE
            hidden_size = self.pad_dim * self.num_heads
            return hidden_size


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
            raise ValueError("Input hidden_states should not be none.")

        # only support BNC now.
        if hidden_states.ndim != 3: # 3: BNC
            raise ValueError(f"The dimensions of hidden_states should be 3, but got {hidden_states.ndim}")

        batch_size = hidden_states.shape[0]
        
        if attn.group_norm is not None:
            # In `BSH`, `H` represents channel, so it needs to be transposed.
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.is_cross_attention:
            query, key, value = attn.qkv_proj(hidden_states, encoder_hidden_states)
        else:
            query, key, value = attn.qkv_proj(hidden_states)
        
        query = query.reshape(batch_size, -1, attn.num_heads, attn.pad_dim).transpose(1, 2)  # BNSD
        key = key.reshape(batch_size, -1, attn.num_heads, attn.pad_dim).transpose(1, 2)  # BNSD
        value = value.reshape(batch_size, -1, attn.num_heads, attn.pad_dim).transpose(1, 2)  # BNSD

        # norm q and k
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # position embedding q and k
        query = attn.position_embedding(query)
        key = attn.position_embedding(key)

        hidden_states = torch_npu.npu_prompt_flash_attention(
            query, key, value,
            num_heads=query.shape[1],
            input_layout="BNSD",
            scale_value=attn.scale_value,
            pre_tokens=MAX_TOKENS,
            next_tokens=MAX_TOKENS,
            sparse_mode=0)

        # transform the hidden_states layout from BNSD to BSH
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.num_heads * attn.pad_dim)
        hidden_states = attn.out_proj(hidden_states)
        return hidden_states


class HunyuanAttnProcessor:
    """
    The Hunyuan attention processor.
    """
    def __call__(
        self,
        attn: ReconstitutionAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        if hidden_states is None:
            raise ValueError("Input hidden_states should not be none.")

        # only support BNC now.
        if hidden_states.ndim != 3: # 3: BNC
            raise ValueError(f"The dimensions of hidden_states should be 3, but got {hidden_states.ndim}")

        batch_size = hidden_states.shape[0]
        
        if attn.is_cross_attention:
            query, key, value = attn.qkv_proj(hidden_states, encoder_hidden_states)
        else:
            query, key, value = attn.qkv_proj(hidden_states)
        
        query = query.reshape(batch_size, -1, attn.num_heads, attn.pad_dim).transpose(1, 2)  # BNSD
        key = key.reshape(batch_size, -1, attn.num_heads, attn.pad_dim).transpose(1, 2)  # BNSD
        value = value.reshape(batch_size, -1, attn.num_heads, attn.pad_dim).transpose(1, 2)  # BNSD
        
        # norm q and k
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # position embedding q and k
        query = attn.position_embedding(query, rotary_emb)
        if not attn.is_cross_attention:
            key = attn.position_embedding(key, rotary_emb)

        # need replaced by dispatch flash_attention function
        hidden_states = torch_npu.npu_fusion_attention(
            query, key, value,
            head_num=attn.num_heads,
            input_layout="BNSD",
            scale=attn.scale_value,
        )[0]

        # transform the hidden_states layout from BNSD to BSH
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.num_heads * attn.pad_dim)
        hidden_states = attn.out_proj(hidden_states)
        return hidden_states