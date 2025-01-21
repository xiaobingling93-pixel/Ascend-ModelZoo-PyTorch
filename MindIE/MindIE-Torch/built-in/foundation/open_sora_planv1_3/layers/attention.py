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

import math
import inspect
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from einops import rearrange, repeat

from .norm import get_normalization_helper
from .embedding import RoPE3D, PositionGetter3D, get_embedding_helper
from ..models.parallel_mgr import get_sequence_parallel_state, get_sequence_parallel_size
from ..models.comm import all_to_all_sbh
from .linear import QKVLinear

ALIGNMENT_BASE = 16


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
            self.add_qkv_proj = nn.Linear(add_proj_dim, 3 * hidden_size, bias=add_proj_bias) # 3: qkv

        # OutLinear
        self.out_proj = nn.Linear(hidden_size, attention_dim, bias=out_proj_bias)
        
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
        attn: ReconstitutionAttention,
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
            query = attn.q_proj(hidden_states)
            query = query.reshape(batch_size, -1, attn.num_heads, attn.head_dim).transpose(1, 2) # B S N D -> B N S D
            
            kv = attn.kv_proj(encoder_hidden_states)
            kv = kv.reshape(batch_size, -1, 2, attn.num_heads, attn.head_dim)
            key, value = kv.permute(2, 0, 3, 1, 4).unbind(0) # B S 2 N D -> 2 B N S D -> 2 * B N S D
        else:
            qkv = attn.qkv_proj(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 3, attn.num_heads, attn.head_dim) # 3: q,k,v
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0) # B S 3 N D -> 3 B N S D -> 3 * B N S D

        hidden_states = torch_npu.npu_prompt_flash_attention(
            query, key, value,
            num_heads=query.shape[1],
            input_layout="BNSD",
            atten_mask=attention_mask,
            scale_value=attn.scale_value)
        # transform the hidden_states layout from BNSD to BSH
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.num_heads * attn.head_dim)
        hidden_states = attn.out_proj(hidden_states)
        return hidden_states


class OpenSoraPlanAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, interpolation_scale_thw=(1, 1, 1), 
                 sparse1d=False, sparse_n=2, sparse_group=False, is_cross_attn=True):
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw
        
        self._init_rope(interpolation_scale_thw)

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8, 
        height: int = 16, 
        width: int = 16, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        _, batch_size, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        
        if attn.is_cross_attention:
            query, key, value = attn.qkv_proj(hidden_states, encoder_hidden_states)
        else:
            query, key, value = attn.qkv_proj(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.num_heads
        fa_head_num = attn.num_heads
        total_frame = frame

        if get_sequence_parallel_state():
            sp_size = get_sequence_parallel_size()
            fa_head_num = attn.num_heads // sp_size
            total_frame = frame * sp_size
            # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
            query = all_to_all_sbh(query.view(-1, attn.num_heads, head_dim), scatter_dim=1, gather_dim=0)
            key = all_to_all_sbh(key.view(-1, attn.num_heads, head_dim), scatter_dim=1, gather_dim=0)
            value = all_to_all_sbh(value.view(-1, attn.num_heads, head_dim), scatter_dim=1, gather_dim=0)
        query = query.view(-1, batch_size, fa_head_num, head_dim)
        key = key.view(-1, batch_size, fa_head_num, head_dim)

        if not self.is_cross_attn:
            # require the shape of (ntokens x batch_size x nheads x dim)
            pos_thw = self.position_getter(batch_size, t=total_frame, h=height, w=width, device=query.device)

            query = self.rope(query, pos_thw)
            key = self.rope(key, pos_thw)

        query = query.view(-1, batch_size, fa_head_num * head_dim)
        key = key.view(-1, batch_size, fa_head_num * head_dim)
        value = value.view(-1, batch_size, fa_head_num * head_dim)
        if self.sparse1d:
            query, pad_len = self._sparse_1d(query, total_frame, height, width)
            if self.is_cross_attn:
                key = self._sparse_1d_kv(key)
                value = self._sparse_1d_kv(value)
            else:
                key, pad_len = self._sparse_1d(key, total_frame, height, width)
                value, pad_len = self._sparse_1d(value, total_frame, height, width)

        rearrange_method = 's b (h d) -> b h s d'
        # .contiguous() not need
        query = rearrange(query, rearrange_method, h=fa_head_num) 
        key = rearrange(key, rearrange_method, h=fa_head_num)
        value = rearrange(value, rearrange_method, h=fa_head_num)

        hidden_states = torch_npu.npu_fused_infer_attention_score(query, key, value,
            atten_mask=attention_mask, input_layout="BNSD", scale=1 / math.sqrt(head_dim),
            num_heads=fa_head_num)[0]

        hidden_states = rearrange(hidden_states, 'b h s d -> s b (h d)', h=fa_head_num).contiguous()

        if self.sparse1d:
            hidden_states = self._reverse_sparse_1d(
                hidden_states, total_frame, height, width, pad_len)

        # [s, b, h // sp * d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
        if get_sequence_parallel_state():
            hidden_states = all_to_all_sbh(hidden_states.reshape(-1, fa_head_num, head_dim),
                                            scatter_dim=0, gather_dim=1)
            hidden_states = hidden_states.view(-1, batch_size, inner_dim)

        hidden_states = hidden_states.to(query.dtype)
        # linear proj
        hidden_states = attn.out_proj(hidden_states)
        return hidden_states

    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()
    
    def _sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        seqlen = x.shape[0]
        if seqlen != frame * height * width:
            raise ValueError(f"x.shape[0] should be equal to frame*height*width")
        pad_len = 0
        if seqlen % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - seqlen % (self.sparse_n * self.sparse_n)
        if pad_len != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        if not self.sparse_group:
            x = rearrange(x, '(g k) b d -> g (k b) d', k=self.sparse_n)
        else:
            x = rearrange(x, '(n m k) b d -> (n k) (m b) d', m=self.sparse_n, k=self.sparse_n)
        return x, pad_len
    
    def _reverse_sparse_1d(self, x, frame, height, width, pad_len):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        if x.shape[0] != (frame * height * width + pad_len) // self.sparse_n:
            raise ValueError("x.shape[0] should be equal to" 
                                    f"f{(frame * height * width + pad_len) // self.sparse_n}")
        if not self.sparse_group:
            x = rearrange(x, 'g (k b) d -> (g k) b d', k=self.sparse_n)
        else:
            x = rearrange(x, '(n k) (m b) d -> (n m k) b d', m=self.sparse_n, k=self.sparse_n)
        x = x[:frame * height * width, :, :]
        return x
    
    def _sparse_1d_kv(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = repeat(x, 's b d -> s (k b) d', k=self.sparse_n)
        return x