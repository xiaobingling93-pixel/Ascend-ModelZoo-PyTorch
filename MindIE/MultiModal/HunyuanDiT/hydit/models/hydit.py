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


from typing import Tuple, List

import torch
import torch.nn as nn

from mindiesd import ConfigMixin, get_activation_layer
from .model_utils import DiffusionModel
from ..layers import get_normalization_helper, timestep_embedding
from ..layers import Mlp, PatchEmbed, TimestepEmbedder, Attention, AttentionPool


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
        rotated_mode = "rotated_interleaved"

        # ========================= Self-Attention =========================
        self.norm1 = get_normalization_helper(norm_type, hidden_size, eps=1e-6)
        self.attn1 = Attention(hidden_size=hidden_size,
                               cross_attention_dim=None,
                               num_heads=num_heads,
                               attention_norm=norm_type,
                               rotated_mode=rotated_mode)

        # ========================= FFN =========================
        self.norm2 = get_normalization_helper(norm_type, hidden_size, eps=1e-6)
        self.mlp = Mlp(
            features_in=hidden_size, features_hidden=int(hidden_size * mlp_ratio), act_layer="gelu-tanh")

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(
            get_activation_layer("silu"),
            nn.Linear(c_emb_size, hidden_size, bias=True)
        )

        # ========================= Cross-Attention =========================
        self.attn2 = Attention(hidden_size=hidden_size,
                               cross_attention_dim=text_states_dim,
                               num_heads=num_heads,
                               attention_norm=norm_type,
                               rotated_mode=rotated_mode)
        self.norm3 = get_normalization_helper(norm_type, hidden_size, eps=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = get_normalization_helper(norm_type, 2 * hidden_size, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None


    def forward(self, x, tensor_input, skip=None, layer=0):
        c, text_states, freqs_cis_img = tensor_input
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)
        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        x = x + self.attn1(hidden_states=self.norm1(x) + shift_msa,
                           freqs_cis_img=freqs_cis_img,
                           layer=layer)
        # Cross-Attention
        x = x + self.attn2(hidden_states=self.norm3(x),
                           encoder_hidden_states=text_states,
                           freqs_cis_img=freqs_cis_img,
                           layer=layer)
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
            get_activation_layer("silu"),
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
        size_cond: List = None,
        use_style_cond: bool = False,
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
        self.size_cond = size_cond
        self.use_style_cond = use_style_cond


class HunyuanDiT2DModel(DiffusionModel):

    config_class = HunyuanDiTConfig
    weigths_name = "pytorch_model_ema.pt"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._check_config_params()

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
        self.pooler = AttentionPool(self.config.text_len_t5,
                                    self.config.text_states_dim_t5,
                                    num_heads=8,
                                    output_dim=pooler_out_dim)

        # Dimension of the extra input vectors
        self.extra_in_dim = pooler_out_dim

        # Only for hydit <= 1.1
        if self.config.size_cond:
            # Image size and crop size conditions
            self.extra_in_dim += 6 * 256
        if self.config.use_style_cond:
            # Here we use a default learned embedder layer for future extension.
            self.style_embedder = nn.Embedding(1, self.config.hidden_size)
            self.extra_in_dim += self.config.hidden_size

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

        x, t, encoder_hidden_states, embeds_and_mask_input, freqs_cis_img = tensor_input
        if use_cache:
            block_start, num_blocks, delta_cache = cache_params

        text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5, image_meta_size, style = \
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

        # The input x shape is [2, 4, 128, 128]
        height, width = x.shape[-2:]
        th, tw = height // self.config.patch_size, width // self.config.patch_size

        # Build time and image embedding
        t = self.t_embedder(t)
        x = self.x_embedder(x)
        # The x shape after x_embedder is [2, 4096, 1408]

        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        # Only for hydit <= 1.1
        if image_meta_size is not None:
            image_meta_size = timestep_embedding(image_meta_size.half().view(-1), 256)   # [B * 6, 256]
            image_meta_size = image_meta_size.half().view(-1, 6 * 256)
            extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)  # [B, D + 6 * 256]
        if style is not None:
            style_embedding = self.style_embedder(style)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # Forward pass through HunYuanDiT blocks
        tensor_input = (c, text_states, freqs_cis_img)
        if not use_cache:
            skips = []
            for layer, block in enumerate(self.blocks):
                if layer > self.config.depth // 2:
                    skip = skips.pop()
                    x = block(x, tensor_input, skip=skip, layer=layer)         # (N, L, D)
                else:
                    x = block(x, tensor_input, skip=None, layer=layer)         # (N, L, D)

                if layer < (self.config.depth // 2 - 1):
                    skips.append(x)
        else:
            cache_params = (use_cache, if_skip, block_start, num_blocks)
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
                x = block(x, tensor_input, skip=skip, layer=layer)         # (N, L, D)
            else:
                x = block(x, tensor_input, skip=None, layer=layer)         # (N, L, D)

            if layer < (self.config.depth // 2 - 1):
                skips.append(x)
        return x, skips


    def _forward_blocks(self, x, tensor_input, cache_params, delta_cache):
        use_cache, if_skip, block_start, num_blocks = cache_params
        skips = []
        if not use_cache:
            x, skips = self._forward_blocks_range(x, tensor_input, skips, 0, len(self.blocks))
        else:
            x, skips = self._forward_blocks_range(x, tensor_input, skips, 0, block_start)

            cache_end = block_start + num_blocks
            x_before_cache = x.clone()
            if not if_skip:
                x, skips = self._forward_blocks_range(x, tensor_input, skips, block_start, cache_end)
                delta_cache = x - x_before_cache
            else:
                x = x_before_cache + delta_cache
            
            x, skips = self._forward_blocks_range(x, tensor_input, skips, cache_end, len(self.blocks))
        return x, delta_cache


    def _load_weights(self, state_dict):
        weights = state_dict

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

            qkv_proj_weights = weights.pop(prefix_key + 'attn1.Wqkv.weight')
            qkv_proj_bias = weights.pop(prefix_key + 'attn1.Wqkv.bias')
            to_q_weights, to_k_weights, to_v_weights = torch.chunk(qkv_proj_weights, 3, dim=0)
            to_q_bias, to_k_bias, to_v_bias = torch.chunk(qkv_proj_bias, 3, dim=0)
            weights[prefix_key + 'attn1.q_proj.weight'] = to_q_weights
            weights[prefix_key + 'attn1.q_proj.bias'] = to_q_bias
            weights[prefix_key + 'attn1.kv_proj.weight'] = torch.cat([to_k_weights, to_v_weights], dim=0)
            weights[prefix_key + 'attn1.kv_proj.bias'] = torch.cat([to_k_bias, to_v_bias], dim=0)

        self.load_state_dict(weights)


    def _unpatchify(self, x, h, w):
        c = self.unpatchify_channels
        p = self.config.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


    def _check_config_params(self):
        params_checks = {
            "patch_size": int,
            "in_channels": int,
            "hidden_size": int,
            "depth": int,
            "num_heads": int,
            "mlp_ratio": float,
            "text_states_dim": int,
            "text_states_dim_t5": int,
            "text_len": int,
            "text_len_t5": int
        }
        for attr, expected_type in params_checks.items():
            if hasattr(self.config, attr) and not isinstance(getattr(self.config, attr), expected_type):
                raise TypeError(f"The type of {attr} in config must be {expected_type.name}, but got {type(attr)}.")
            if getattr(self.config, attr) <= 0:
                raise ValueError(f"The {attr} in config must be greater than 0, but got {attr}.")
        if self.config.hidden_size < self.config.num_heads:
            raise ValueError(f"The hidden_size must be greater than num_heads.")