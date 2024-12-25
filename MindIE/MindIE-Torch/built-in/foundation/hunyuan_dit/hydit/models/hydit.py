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


from typing import Tuple
import math

import torch
import torch.nn as nn
import torch_npu

from mindiesd import ConfigMixin, DiffusionModel
from ..layers.attention import ReconstitutionAttention, HunyuanAttnProcessor
from ..layers.mlp import Mlp
from ..layers.embedding import PatchEmbed, TimestepEmbedder
from ..layers.norm import get_normalization_helper
from ..layers.activation import get_activation_fn

MAX_TOKENS = 2147483647


class HunyuanAttentionPool(nn.Module):
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.attn = ReconstitutionAttention(
            attention_dim=embed_dim,
            cross_attention_dim=embed_dim,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            qkv_bias=True,
            out_proj_bias=True,
            out_dim=output_dim,
        )


    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        tgt_len, bsz, embed_dim = x[:1].size()
        src_len = x.size(0)

        query, key, value = self.attn.qkv_proj(x[:1], x)

        query = query.reshape(tgt_len, bsz, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        key = key.reshape(src_len, bsz, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        value = value.reshape(src_len, bsz, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)

        scale = self.attn.scale_value
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        output = torch_npu.npu_fusion_attention(
            query, key, value, query.shape[1], input_layout="BNSD",
            pse=None,
            scale=scale_factor,
            pre_tockens=MAX_TOKENS,
            next_tockens=MAX_TOKENS,
            keep_prob=1.,
            sync=False,
            inner_precise=0
        )[0]

        output = output.reshape(tgt_len, bsz, embed_dim)
        output = self.attn.out_proj(output)
        return output.squeeze(0)


class HunyuanDiTBlock(nn.Module):
    """
    A HunYuanDiT block with `add` conditioning.
    """
    def __init__(self,
                 hidden_size,
                 c_emb_size,
                 num_heads,
                 mlp_ratio=4.0,
                 text_states_dim=1024,
                 skip=False,
                 ):
        super().__init__()

        norm_type = "layer_norm"

        # ========================= Self-Attention =========================
        self.norm1 = get_normalization_helper(norm_type, hidden_size, eps=1e-6)

        self.attn1 = ReconstitutionAttention(
            attention_dim=hidden_size,
            cross_attention_dim=None,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=True,
            out_proj_bias=True,
            attention_norm=norm_type,
            position_embedding='rope',
            eps=1e-6,
            processor=HunyuanAttnProcessor(),
        )

        # ========================= FFN =========================
        self.norm2 = get_normalization_helper(norm_type, hidden_size, eps=1e-6)
        self.mlp = Mlp(
            features_in=hidden_size, features_hidden=int(hidden_size * mlp_ratio), act_layer="gelu-approximate")

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(
            get_activation_fn("silu"),
            nn.Linear(c_emb_size, hidden_size, bias=True)
        )

        # ========================= Cross-Attention =========================
        self.attn2 = ReconstitutionAttention(
            attention_dim=hidden_size,
            cross_attention_dim=text_states_dim,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=True,
            out_proj_bias=True,
            attention_norm=norm_type,
            position_embedding='rope',
            eps=1e-6,
            processor=HunyuanAttnProcessor(),
        )

        self.norm3 = get_normalization_helper(norm_type, hidden_size, eps=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = get_normalization_helper(norm_type, 2 * hidden_size, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None


    def forward(self, x, tensor_input, skip=None):
        c, text_states, freqs_cis_img = tensor_input
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        x = x + self.attn1(hidden_states=self.norm1(x) + shift_msa,
                           rotary_emb=freqs_cis_img)

        # Cross-Attention
        x = x + self.attn2(hidden_states=self.norm3(x),
                           encoder_hidden_states=text_states,
                           rotary_emb=freqs_cis_img)

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """
    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            get_activation_fn("silu"),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class HunyuanDiTConfig(ConfigMixin):
    config_name = 'config.json'

    def __init__(
        self,
        input_size: Tuple[int, int] = (None, None),
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        text_states_dim: int = 1024,
        text_states_dim_t5: int = 2048,
        text_len: int = 77,
        text_len_t5: int = 256,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.text_states_dim = text_states_dim
        self.text_states_dim_t5 = text_states_dim_t5
        self.text_len = text_len
        self.text_len_t5 = text_len_t5


class HunyuanDiT2DModel(DiffusionModel):

    config_class = HunyuanDiTConfig
    weigths_name = "pytorch_model_ema.pt"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # learn_sigma is True
        self.out_channels = self.config.in_channels * 2

        self.mlp_t5 = Mlp(features_in=self.config.text_states_dim_t5,
                          features_hidden=self.config.text_states_dim_t5 * 4,
                          features_out=self.config.text_states_dim,
                          act_layer="silu",
                          bias=True)

        # learnable replace
        self.text_embedding_padding = nn.Parameter(
            torch.randn(self.config.text_len + self.config.text_len_t5,
                        self.config.text_states_dim,
                        dtype=torch.float32))

        # Attention pooling
        pooler_out_dim = 1024
        self.pooler = HunyuanAttentionPool(self.config.text_len_t5, 
                                           self.config.text_states_dim_t5, 
                                           num_heads=8, 
                                           output_dim=pooler_out_dim)

        # Dimension of the extra input vectors
        self.extra_in_dim = pooler_out_dim

        # Text embedding for `add`
        height = self.config.input_size[0] // 8
        width = self.config.input_size[1] // 8
        self.x_embedder = PatchEmbed(height,
                                     width,
                                     self.config.patch_size,
                                     self.config.in_channels,
                                     self.config.hidden_size,
                                     pos_embed_type=None)

        self.t_embedder = TimestepEmbedder(self.config.hidden_size)
        self.extra_embedder = Mlp(features_in=self.extra_in_dim,
                                  features_hidden=self.config.hidden_size * 4,
                                  features_out=self.config.hidden_size,
                                  act_layer="silu",
                                  bias=True)

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList([
            HunyuanDiTBlock(hidden_size=self.config.hidden_size,
                            c_emb_size=self.config.hidden_size,
                            num_heads=self.config.num_heads,
                            mlp_ratio=self.config.mlp_ratio,
                            text_states_dim=self.config.text_states_dim,
                            skip=layer > self.config.depth // 2)
            for layer in range(self.config.depth)
        ])
        self.final_layer = FinalLayer(self.config.hidden_size,
                                      self.config.hidden_size,
                                      self.config.patch_size,
                                      self.out_channels)
        self.unpatchify_channels = self.out_channels


    def forward(self,
                tensor_input=None,
                use_cache: bool = False,
                cache_params=None,
                if_skip: int = 0):

        x, t, embeds_and_mask_input, freqs_cis_img = tensor_input
        if use_cache:
            cache_dict, delta_cache = cache_params

        encoder_hidden_states, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5 = \
            embeds_and_mask_input
        text_states = encoder_hidden_states
        text_states_t5 = encoder_hidden_states_t5
        text_states_mask = text_embedding_mask.bool()
        text_states_t5_mask = text_embedding_mask_t5.bool()
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states = torch.cat([text_states, text_states_t5.view(b_t5, l_t5, -1)], dim=1)  # 2,205，1024
        clip_t5_mask = torch.cat([text_states_mask, text_states_t5_mask], dim=-1)

        clip_t5_mask = clip_t5_mask
        text_states = torch.where(clip_t5_mask.unsqueeze(2), text_states, self.text_embedding_padding.to(text_states))

        height, width = x.shape[-2:]
        th, tw = height // self.config.patch_size, width // self.config.patch_size

        # Build time and image embedding
        t = self.t_embedder(t, t.dtype)
        x = self.x_embedder(x)

        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # Forward pass through HunYuanDiT blocks
        tensor_input = (c, text_states, freqs_cis_img)
        if not use_cache:
            skips = []
            for layer, block in enumerate(self.blocks):
                if layer > self.config.depth // 2:
                    skip = skips.pop()
                    x = block(x, tensor_input, skip)   # (N, L, D)
                else:
                    x = block(x, tensor_input)         # (N, L, D)

                if layer < (self.config.depth // 2 - 1):
                    skips.append(x)
        else:
            cache_params = (use_cache, if_skip, cache_dict)
            x, delta_cache = self._forward_blocks(x, tensor_input, cache_params, delta_cache)

        # Final layer
        x = self.final_layer(x, c)                              # (N, L, patch_size ** 2 * out_channels)
        x = self._unpatchify(x, th, tw)

        if use_cache:
            return x, delta_cache

        return x


    def _forward_blocks_range(self, x, tensor_input, skips, start_idx, end_idx):
        for layer, block in zip(range(start_idx, end_idx), self.blocks[start_idx : end_idx]):
            if layer > self.config.depth // 2:
                skip = skips.pop()
                x = block(x, tensor_input, skip)   # (N, L, D)
            else:
                x = block(x, tensor_input)         # (N, L, D)

            if layer < (self.config.depth // 2 - 1):
                skips.append(x)
        return x, skips
    

    def _forward_blocks(self, x, tensor_input, cache_params, delta_cache):
        use_cache, if_skip, cache_dict = cache_params
        skips = []
        if not use_cache:
            x, skips = self._forward_blocks_range(x, tensor_input, skips, 0, len(self.blocks))
        else:
            x, skips = self._forward_blocks_range(x, tensor_input, skips, 0, cache_dict[0])

            cache_end = cache_dict[0] + cache_dict[2]
            x_before_cache = x.clone()
            if not if_skip:
                x, skips = self._forward_blocks_range(x, tensor_input, skips, cache_dict[0], cache_end)
                delta_cache = x - x_before_cache
            else:
                x = x_before_cache + delta_cache
            
            x, skips = self._forward_blocks_range(x, tensor_input, skips, cache_end, len(self.blocks))
        return x, delta_cache


    def _load_weights(self, state_dict):
        weights = state_dict

        weights['pooler.attn.qkv_proj.q_weight'] = weights.pop(
            'pooler.q_proj.weight').transpose(0, 1).contiguous()
        weights['pooler.attn.qkv_proj.q_bias'] = weights.pop('pooler.q_proj.bias')
        to_k_weights = weights.pop('pooler.k_proj.weight')
        to_k_bias = weights.pop('pooler.k_proj.bias')
        to_v_weights = weights.pop('pooler.v_proj.weight')
        to_v_bias = weights.pop('pooler.v_proj.bias')
        weights['pooler.attn.qkv_proj.kv_weight'] = torch.cat(
            [to_k_weights, to_v_weights], dim=0).transpose(0, 1).contiguous()
        weights['pooler.attn.qkv_proj.kv_bias'] = torch.cat([to_k_bias, to_v_bias], dim=0)
        weights['pooler.attn.out_proj.weight'] = weights.pop('pooler.c_proj.weight')
        weights['pooler.attn.out_proj.bias'] = weights.pop('pooler.c_proj.bias')

        weights['mlp_t5.fc1.weight'] = weights.pop('mlp_t5.0.weight')
        weights['mlp_t5.fc1.bias'] = weights.pop('mlp_t5.0.bias')
        weights['mlp_t5.fc2.weight'] = weights.pop('mlp_t5.2.weight')
        weights['mlp_t5.fc2.bias'] = weights.pop('mlp_t5.2.bias')

        weights['extra_embedder.fc1.weight'] = weights.pop('extra_embedder.0.weight')
        weights['extra_embedder.fc1.bias'] = weights.pop('extra_embedder.0.bias')
        weights['extra_embedder.fc2.weight'] = weights.pop('extra_embedder.2.weight')
        weights['extra_embedder.fc2.bias'] = weights.pop('extra_embedder.2.bias')

        for i in range(self.config.depth):
            prefix_key = 'blocks.' + str(i) + '.'
            weights[prefix_key + 'norm1.weight'] = weights.pop(prefix_key + 'norm1.weight')
            weights[prefix_key + 'norm1.bias'] = weights.pop(prefix_key + 'norm1.bias')

            weights[prefix_key + 'attn1.qkv_proj.weight'] = weights.pop(
                prefix_key + 'attn1.Wqkv.weight').transpose(0, 1).contiguous()
            weights[prefix_key + 'attn1.qkv_proj.bias'] = weights.pop(prefix_key + 'attn1.Wqkv.bias')

            weights[prefix_key + 'attn1.norm_q.weight'] = weights.pop(prefix_key + 'attn1.q_norm.weight')
            weights[prefix_key + 'attn1.norm_q.bias'] = weights.pop(prefix_key + 'attn1.q_norm.bias')
            weights[prefix_key + 'attn1.norm_k.weight'] = weights.pop(prefix_key + 'attn1.k_norm.weight')
            weights[prefix_key + 'attn1.norm_k.bias'] = weights.pop(prefix_key + 'attn1.k_norm.bias')

            weights[prefix_key + 'attn1.out_proj.weight'] = weights.pop(prefix_key + 'attn1.out_proj.weight')
            weights[prefix_key + 'attn1.out_proj.bias'] = weights.pop(prefix_key + 'attn1.out_proj.bias')

            weights[prefix_key + 'norm2.weight'] = weights.pop(prefix_key + 'norm2.weight')
            weights[prefix_key + 'norm2.bias'] = weights.pop(prefix_key + 'norm2.bias')

            weights[prefix_key + 'mlp.fc1.weight'] = weights.pop(prefix_key + 'mlp.fc1.weight')
            weights[prefix_key + 'mlp.fc1.bias'] = weights.pop(prefix_key + 'mlp.fc1.bias')

            weights[prefix_key + 'mlp.fc2.weight'] = weights.pop(prefix_key + 'mlp.fc2.weight')
            weights[prefix_key + 'mlp.fc2.bias'] = weights.pop(prefix_key + 'mlp.fc2.bias')

            weights[prefix_key + 'default_modulation.1.weight'] = weights.pop(
                prefix_key + 'default_modulation.1.weight')
            weights[prefix_key + 'default_modulation.1.bias'] = weights.pop(
                prefix_key + 'default_modulation.1.bias')

            weights[prefix_key + 'attn2.qkv_proj.q_weight'] = weights.pop(
                prefix_key + 'attn2.q_proj.weight').transpose(0, 1).contiguous()
            weights[prefix_key + 'attn2.qkv_proj.q_bias'] = weights.pop(prefix_key + 'attn2.q_proj.bias')

            weights[prefix_key + 'attn2.qkv_proj.kv_weight'] = weights.pop(
                prefix_key + 'attn2.kv_proj.weight').transpose(0, 1).contiguous()
            weights[prefix_key + 'attn2.qkv_proj.kv_bias'] = weights.pop(prefix_key + 'attn2.kv_proj.bias')

            weights[prefix_key + 'attn2.norm_q.weight'] = weights.pop(prefix_key + 'attn2.q_norm.weight')
            weights[prefix_key + 'attn2.norm_q.bias'] = weights.pop(prefix_key + 'attn2.q_norm.bias')
            weights[prefix_key + 'attn2.norm_k.weight'] = weights.pop(prefix_key + 'attn2.k_norm.weight')
            weights[prefix_key + 'attn2.norm_k.bias'] = weights.pop(prefix_key + 'attn2.k_norm.bias')

            weights[prefix_key + 'attn2.out_proj.weight'] = weights.pop(prefix_key + 'attn2.out_proj.weight')
            weights[prefix_key + 'attn2.out_proj.bias'] = weights.pop(prefix_key + 'attn2.out_proj.bias')

            weights[prefix_key + 'norm3.weight'] = weights.pop(prefix_key + 'norm3.weight')
            weights[prefix_key + 'norm3.bias'] = weights.pop(prefix_key + 'norm3.bias')

            if i > self.config.depth // 2:
                weights[prefix_key + 'skip_norm.weight'] = weights.pop(prefix_key + 'skip_norm.weight')
                weights[prefix_key + 'skip_norm.bias'] = weights.pop(prefix_key + 'skip_norm.bias')
                weights[prefix_key + 'skip_linear.weight'] = weights.pop(prefix_key + 'skip_linear.weight')
                weights[prefix_key + 'skip_linear.bias'] = weights.pop(prefix_key + 'skip_linear.bias')

        self.load_state_dict(weights)


    def _unpatchify(self, x, h, w):
        c = self.unpatchify_channels
        p = self.config.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs