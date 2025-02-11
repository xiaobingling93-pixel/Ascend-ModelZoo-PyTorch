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

import os
from typing import Any, Dict, Optional, Tuple
import inspect

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mindiesd import ConfigMixin
from mindiesd import DiffusionModel

from ..layers import Mlp, AdaLayerNorm
from ..layers.embedding import PatchEmbed2D
from ..layers.embedding import AdaLayerNormSingle
from ..layers.attention import ReconstitutionAttention, OpenSoraPlanAttnProcessor
from .model_utils import get_attn_weight, weight_switch


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1), 
        sparse1d: bool = False,
        sparse_n: int = 2,
        sparse_group: bool = False,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = AdaLayerNorm(dim, norm_eps)

        processor = OpenSoraPlanAttnProcessor(
            interpolation_scale_thw=interpolation_scale_thw, sparse1d=sparse1d, sparse_n=sparse_n, 
            sparse_group=sparse_group, is_cross_attn=False
            )
        self.attn1 = ReconstitutionAttention(
            attention_dim=dim,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qkv_bias=attention_bias,
            out_proj_bias=attention_out_bias,
            processor=processor
        )  # is self-attn if encoder_hidden_states is none

        # 2. Cross-Attn
        self.norm2 = AdaLayerNorm(dim, norm_eps)

        processor = OpenSoraPlanAttnProcessor(
            interpolation_scale_thw=interpolation_scale_thw, sparse1d=sparse1d, sparse_n=sparse_n, 
            sparse_group=sparse_group, is_cross_attn=True
            )
        self.attn2 = ReconstitutionAttention(
            attention_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qkv_bias=attention_bias,
            out_proj_bias=attention_out_bias,
            processor=processor
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        ff_inner_dim = ff_inner_dim or 4 * dim
        self.ff = Mlp(features_in=dim, features_hidden=ff_inner_dim, 
                      act_layer=activation_fn, bias=ff_bias)

        # 4. Scale-shift.
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
    ) -> torch.FloatTensor:
        
        # 0. Self-Attention
        batch_size = hidden_states.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
        ).chunk(6, dim=0)

        norm_hidden_states = self.norm1(hidden_states, shift_msa, (1 + scale_msa))
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask, frame=frame, height=height, width=width, 
        )

        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        norm_hidden_states = hidden_states

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask, frame=frame, height=height, width=width,
        )
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states, shift_mlp, (1 + scale_mlp))

        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        return hidden_states
    

class OpenSoraT2Vv1_3Config(ConfigMixin):
    config_name = 'config.json'

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = True,
        sample_size_h: Optional[int] = None,
        sample_size_w: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        interpolation_scale_h: float = 1.0,
        interpolation_scale_w: float = 1.0,
        interpolation_scale_t: float = 1.0,
        sparse1d: bool = False,
        sparse_n: int = 2,
    ):
        self._init(locals())

    def _init(self, value):
        init_signature = inspect.signature(self.__init__)
        parameters = init_signature.parameters
        for param_name, _ in parameters.items():
            if param_name != 'self':
                setattr(self, param_name, value[param_name])  


class OpenSoraT2Vv1_3(DiffusionModel):
    config_class = OpenSoraT2Vv1_3Config
    weigths_name = "diffusion_pytorch_model.safetensors.index.json"
    
    def __init__(self, config):
        super().__init__(config)
        # Set some common variables used across the board.
        self.out_channels = config.in_channels if config.out_channels is None else config.out_channels
        self.config.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        self.use_cache = False
        self.cache = None
        self._prepare_patched_inputs()
        
    def load_weights(self, state_dict):
        with torch.no_grad():
            weights = state_dict
            # attention_block:
            cache_weights = {}
            for i in range(self.config.num_layers):
                
                prefix_key = 'transformer_blocks.' + str(i) + '.'
                cache_weights1 = get_attn_weight(weights, prefix_key + "attn1.", cross_attention=False)
                cache_weights2 = get_attn_weight(weights, prefix_key + "attn2.", cross_attention=True)
                cache_weights.update(cache_weights1)
                cache_weights.update(cache_weights2)

                prefix_key = prefix_key + 'ff.'
                weight_switch(weights, prefix_key, 'fc1.weight', 'net.0.proj.weight')
                weight_switch(weights, prefix_key, 'fc1.bias', 'net.0.proj.bias')
                weight_switch(weights, prefix_key, 'fc2.weight', 'net.2.weight')
                weight_switch(weights, prefix_key, 'fc2.bias', 'net.2.bias')

            prefix_key = "caption_projection."
            weight_switch(weights, prefix_key, 'fc1.weight', 'linear_1.weight')
            weight_switch(weights, prefix_key, 'fc1.bias', 'linear_1.bias')
            weight_switch(weights, prefix_key, 'fc2.weight', 'linear_2.weight')
            weight_switch(weights, prefix_key, 'fc2.bias', 'linear_2.bias')
            
            self.load_state_dict(state_dict, strict=False)
            return state_dict.keys(), cache_weights
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        step_id: int = 0,
        **kwargs, 
    ):
        batch_size, _, frame, h, w = hidden_states.shape
        attention_mask, encoder_attention_mask = self._standard_mask(attention_mask, encoder_attention_mask)
        # 1. Input
        frame = ((frame - 1) // self.config.patch_size_t + 1
                 ) if frame % 2 == 1 else frame // self.config.patch_size_t  # patchfy
        height = hidden_states.shape[-2] // self.config.patch_size
        width = hidden_states.shape[-1] // self.config.patch_size

        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, batch_size, frame)

        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

        sparse_mask = {}
        for sparse_n in [1, 4]:
            sparse_mask[sparse_n] = prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n)
            
        # 2. Blocks
        for i, block in enumerate(self.transformer_blocks):
            if i > 1 and i < 30:
                mask_group = sparse_mask.get(block.attn1.processor.sparse_n, None)
                attention_mask, encoder_attention_mask = mask_group.get(block.attn1.processor.sparse_group, None)
            else:
                mask_group = sparse_mask.get(1, None)
                attention_mask, encoder_attention_mask = mask_group.get(block.attn1.processor.sparse_group, None)
            if self.use_cache:
                hidden_states = self.cache(block, step_id, i, hidden_states, 
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep, frame=frame, height=height, width=width, 
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep, frame=frame, height=height, width=width, 
                )
        # New shape (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        video_size = (frame, height, width)
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            embedded_timestep=embedded_timestep,
            video_size=video_size
        )  # b c t h w
        return (output,)

    def _prepare_patched_inputs(self):
        self.config.sample_size = (self.config.sample_size_h, self.config.sample_size_w)
        interpolation_scale_thw = (
            self.config.interpolation_scale_t, 
            self.config.interpolation_scale_h, 
            self.config.interpolation_scale_w
            )
        
        self.caption_projection = Mlp(
            features_in=self.config.caption_channels, 
            features_hidden=self.config.hidden_size,
            features_out=self.config.hidden_size,
            act_layer="gelu-approximate"
        )
        self.pos_embed = PatchEmbed2D(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.hidden_size,
        )
        
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    interpolation_scale_thw=interpolation_scale_thw, 
                    sparse1d=self.config.sparse1d if i > 1 and i < 30 else False, 
                    sparse_n=self.config.sparse_n, 
                    sparse_group=i % 2 == 1, 
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.config.hidden_size) / self.config.hidden_size**0.5)
        self.proj_out = nn.Linear(
            self.config.hidden_size, 
            self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
        )
        self.adaln_single = AdaLayerNormSingle(self.config.hidden_size)

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, batch_size, frame):
        
        hidden_states = self.pos_embed(hidden_states.to(self.dtype))

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d or b, 1, l, d
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep
    
    def _get_output_for_patched_inputs(
        self, hidden_states, embedded_timestep, video_size
    ):  
        (num_frames, height, width) = video_size

        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, num_frames, height, width, 
                   self.config.patch_size_t, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, num_frames * self.config.patch_size_t,
                    height * self.config.patch_size, width * self.config.patch_size)
        )
        return output
    
    def _standard_mask(self, attention_mask, encoder_attention_mask):
        if attention_mask is not None and attention_mask.ndim == 4:

            attention_mask = attention_mask.to(self.dtype)

            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = F.max_pool3d(
                attention_mask, 
                kernel_size=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size), 
                stride=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size)
                )
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> (b 1) 1 (t h w)') 
            attention_mask = (1 - attention_mask.bool().to(self.dtype)) * -10000.0


        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1, l
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
        return attention_mask, encoder_attention_mask


def prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n):
    attention_mask = attention_mask.unsqueeze(1)
    encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    seqlen = attention_mask.shape[-1]
    if seqlen % (sparse_n * sparse_n) == 0:
        pad_len = 0
    else:
        pad_len = sparse_n * sparse_n - seqlen % (sparse_n * sparse_n)
    if pad_len != 0:
        attention_mask_sparse = F.pad(attention_mask, (0, pad_len, 0, 0), value=-9980.0)
        seqlen = attention_mask_sparse.shape[-1]
        attention_mask_sparse_1d = rearrange(
            attention_mask_sparse, 
            'b 1 1 (g k) -> (k b) 1 1 g', 
            k=sparse_n
            )
        attention_mask_sparse_1d_group = rearrange(
            attention_mask_sparse, 
            'b 1 1 (n m k) -> (m b) 1 1 (n k)',
            m=sparse_n, 
            k=sparse_n
            )

    encoder_attention_mask_sparse = encoder_attention_mask.repeat(sparse_n, 1, 1, 1)
    
    encoder_attention_mask_sparse_1d = get_attention_mask(
        encoder_attention_mask_sparse, int(seqlen / sparse_n)
        )
    encoder_attention_mask_sparse_1d_group = encoder_attention_mask_sparse_1d
    if pad_len != 0:
        attention_mask_sparse_1d = get_attention_mask(
            attention_mask_sparse_1d, attention_mask_sparse_1d.shape[-1])
        attention_mask_sparse_1d_group = get_attention_mask(
            attention_mask_sparse_1d_group, attention_mask_sparse_1d_group.shape[-1])
    else:
        attention_mask_sparse_1d = None
        attention_mask_sparse_1d_group = None

    return {
                False: (attention_mask_sparse_1d, encoder_attention_mask_sparse_1d),
                True: (attention_mask_sparse_1d_group, encoder_attention_mask_sparse_1d_group)
            }


def get_attention_mask(attention_mask, repeat_num):
    attention_mask = attention_mask.to(torch.bool)
    attention_mask = attention_mask.repeat_interleave(repeat_num, dim=-2)
    return attention_mask