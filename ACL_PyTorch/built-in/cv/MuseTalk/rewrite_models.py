# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import math
from typing import Callable, Optional, Union
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
import torch
import torch.nn.functional as F
import torch_npu


class RewriteAttnProcessor2_0(AttnProcessor2_0):
    r"""
    Rewrite AttnProcessor2_0: replaced `F.scaled_dot_product_attention` by `torch_npu.npu_prompt_flash_attention`
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # the output = (batch, seq_len, H)
        hidden_states = torch_npu.npu_prompt_flash_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            atten_mask=attention_mask,
            num_heads=attn.heads,
            scale_value=1 / math.sqrt(head_dim),
            input_layout="BSH"
        )

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def rewrite_BasicTransformerBlock(model):
    model.attn1.set_processor(RewriteAttnProcessor2_0())
    model.attn2.set_processor(RewriteAttnProcessor2_0())


def rewrite_Transformer2DModel(model):
    for module in model.transformer_blocks:
        rewrite_BasicTransformerBlock(module)


def rewrite_CrocessAttnBlock2D(model):
    for module in model.attentions:
        rewrite_Transformer2DModel(module)


def rewirte_Unet(model):
    r"""
    Replace AttnProcessor for each sub module: modified flash attention has better performance
    """
    for module in model.down_blocks:
        if "CrossAttnDownBlock2D" in module.__class__.__name__:
            rewrite_CrocessAttnBlock2D(module)
    for module in model.up_blocks:
        if "CrossAttnUpBlock2D" in module.__class__.__name__:
            rewrite_CrocessAttnBlock2D(module)
    rewrite_CrocessAttnBlock2D(model.mid_block)


def rewrite_VAE(model):
    r"""
    Replace AttnProcessor for each sub module: modified flash attention has better performance
    """
    model.encoder.mid_block.attentions[0].set_processor(RewriteAttnProcessor2_0())
    model.decoder.mid_block.attentions[0].set_processor(RewriteAttnProcessor2_0())