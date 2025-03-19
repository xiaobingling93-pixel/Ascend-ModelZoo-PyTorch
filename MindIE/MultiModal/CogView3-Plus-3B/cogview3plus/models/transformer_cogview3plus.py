# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, Union

import torch
import torch.nn as nn
import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.utils import logging
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .modeling_utils import ModelMixin
from .attention import FeedForward
from .attention_processor import CogVideoXAttnProcessor2_0, Attention
from ..layers import CogView3PlusAdaLayerNormZeroTextImage, AdaLayerNormContinuous
from ..layers import CogView3CombinedTimestepSizeEmbeddings, CogView3PlusPatchEmbed


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogView3PlusTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
        useagb: bool = True,
        pab: int = 2,
        total_step: int = 50
    ):
        super().__init__()
        self.useagb = useagb
        self.pab = pab
        self.total_step = total_step

        self.attn_count = 0
        self.last_attn_x_image = None
        self.last_attn_x_prompt = None
        self.attn_alpha_image = 0
        self.attn_alpha_prompt = 0
        self.last_attn_image = None
        self.last_attn_prompt = None
        self.last_ff_image = None
        self.last_ff_prompt = None

        self.norm1 = CogView3PlusAdaLayerNormZeroTextImage(embedding_dim=time_embed_dim, dim=dim)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-6,
            processor=CogVideoXAttnProcessor2_0(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        
        if self.useagb:
            if self.attn_count > 0:
                diff_x_image = hidden_states - self.last_attn_x_image
                diff_x_prompt = encoder_hidden_states - self.last_attn_x_prompt
            
            self.last_attn_x_image = hidden_states
            self.last_attn_x_prompt = encoder_hidden_states

            lower_bound = int(self.total_step / 5) - 0.5
            upper_bound = self.total_step - 1.5
            if (self.attn_count % self.pab != 0) and (lower_bound < self.attn_count < upper_bound):
                broadcast_attn = 1
            else:
                broadcast_attn = 0
            
            if broadcast_attn == 1:
                attn_hidden_states = self.last_attn_image + self.attn_alpha_image * diff_x_image
                attn_encoder_hidden_states = self.last_attn_prompt + self.attn_alpha_prompt * diff_x_prompt
            else:
                # norm & modulate
                norm_hidden_states, chunk_params = self.norm1(hidden_states, encoder_hidden_states, emb)

                gate_msa = chunk_params.gate_msa
                shift_mlp = chunk_params.shift_mlp
                scale_mlp = chunk_params.scale_mlp
                gate_mlp = chunk_params.gate_mlp
                norm_encoder_hidden_states = chunk_params.context
                c_gate_msa = chunk_params.c_gate_msa
                c_shift_mlp = chunk_params.c_shift_mlp
                c_scale_mlp = chunk_params.c_scale_mlp
                c_gate_mlp = chunk_params.c_gate_mlp

                # attention
                attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                    hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
                )

                attn_hidden_states = gate_msa.unsqueeze(1) * attn_hidden_states
                attn_encoder_hidden_states = c_gate_msa.unsqueeze(1) * attn_encoder_hidden_states
                
                # calculate alpha
                if lower_bound < self.attn_count < upper_bound:
                    diff_image = attn_hidden_states - self.last_attn_image
                    diff_prompt = attn_encoder_hidden_states - self.last_attn_prompt

                    self.attn_alpha_image = ((diff_x_image / 100) * (diff_image / 100)).sum() / \
                        ((diff_x_image / 100) ** 2).sum()
                    self.attn_alpha_prompt = ((diff_x_prompt / 100) * (diff_prompt / 100)).sum() / \
                        ((diff_x_prompt / 100) ** 2).sum()
                else:
                    self.attn_alpha_image = 0
                    self.attn_alpha_prompt = 0
                
                self.last_attn_image = attn_hidden_states
                self.last_attn_prompt = attn_encoder_hidden_states
            
            hidden_states = hidden_states + attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states

            if broadcast_attn == 1:
                hidden_states = hidden_states + self.last_ff_image
                encoder_hidden_states = encoder_hidden_states + self.last_ff_prompt
            else:
                # norm & modulate
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + \
                    c_shift_mlp[:, None]

                # feed-forward
                norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
                ff_output = self.ff(norm_hidden_states)

                ff_image = gate_mlp.unsqueeze(1) * ff_output[:, text_seq_length:]
                ff_prompt = c_gate_mlp.unsqueeze(1) * ff_output[:, :text_seq_length]

                hidden_states = hidden_states + ff_image
                encoder_hidden_states = encoder_hidden_states + ff_prompt

                self.last_ff_image = ff_image
                self.last_ff_prompt = ff_prompt
            
            # 更新self.attn_count
            self.attn_count = (self.attn_count + 1) % self.total_step
        else:
            # norm & modulate
            norm_hidden_states, chunk_params = self.norm1(hidden_states, encoder_hidden_states, emb)

            gate_msa = chunk_params.gate_msa
            shift_mlp = chunk_params.shift_mlp
            scale_mlp = chunk_params.scale_mlp
            gate_mlp = chunk_params.gate_mlp
            norm_encoder_hidden_states = chunk_params.context
            c_gate_msa = chunk_params.c_gate_msa
            c_shift_mlp = chunk_params.c_shift_mlp
            c_scale_mlp = chunk_params.c_scale_mlp
            c_gate_mlp = chunk_params.c_gate_mlp

            # attention
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
            )

            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * attn_encoder_hidden_states

            # norm & modulate
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ff_output[:, :text_seq_length]

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states


class CogView3PlusTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        out_channels: int = 16,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        pos_embed_max_size: int = 128,
        use_cache: bool = False,
        cache_interval: int = 2,
        cache_start: int = 1,
        num_cache_layer: int = 11,
        cache_start_steps: int = 10,
        useagb: bool = True,
        pab: int = 2,
        total_step: int = 50,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_layers = num_layers

        # CogView3 uses 3 additional SDXL-like conditions - original_size, target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        self.pooled_projection_dim = 3 * 2 * condition_dim

        self.patch_embed = CogView3PlusPatchEmbed(
            in_channels=in_channels,
            hidden_size=self.inner_dim,
            patch_size=patch_size,
            text_hidden_size=text_embed_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.time_condition_embed = CogView3CombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=self.pooled_projection_dim,
            timesteps_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogView3PlusTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    useagb=useagb,
                    pab=pab,
                    total_step=total_step
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=time_embed_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.q_weight_cache = None
        self.q_bias_cache = None
        self.k_weight_cache = None
        self.k_bias_cache = None
        self.v_weight_cache = None
        self.v_bias_cache = None

        self.use_cache = use_cache
        self.cache_interval = cache_interval
        self.cache_start = cache_start
        self.num_cache_layer = num_cache_layer
        self.cache_start_steps = cache_start_steps

        self.delta_cache = None
        self.delta_encoder_cache = None

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        states,
        timestep: torch.LongTensor,
        original_size: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        hidden_states = states[0]
        encoder_hidden_states = states[1]
        height, width = hidden_states.shape[-2:]
        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = self.patch_embed(
            hidden_states, encoder_hidden_states
        )  # takes care of adding positional embeddings too.
        emb = self.time_condition_embed(timestep, original_size, target_size, crop_coords, hidden_states.dtype)

        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        hidden_states, encoder_hidden_states = self._forward_blocks(hidden_states, encoder_hidden_states, emb, states[2])

        hidden_states = self.norm_out(hidden_states, emb)
        hidden_states = self.proj_out(hidden_states)  # (batch_size, height*width, patch_size*patch_size*out_channels)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, self.out_channels, patch_size, patch_size)
        )
        hidden_states = torch.einsum("nhwcpq->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        return Transformer2DModelOutput(sample=output)

    # forward blocks in range [start_idx, end_idx), then return input and output
    def _forward_blocks_range(self, hidden_states, encoder_hidden_states, emb, start_idx, end_idx, **kwargs):
        for _, block in enumerate(self.transformer_blocks[start_idx: end_idx]):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                emb=emb,
            )

        return hidden_states, encoder_hidden_states

    def _forward_blocks(self, hidden_states, encoder_hidden_states, emb, t_idx):
        num_blocks = len(self.transformer_blocks)

        if not self.use_cache or (t_idx < self.cache_start_steps):
            hidden_states, encoder_hidden_states = self._forward_blocks_range(
                hidden_states, 
                encoder_hidden_states, 
                emb, 
                0, 
                num_blocks
            )
        else:
            # infer [0, cache_start)
            hidden_states, encoder_hidden_states = self._forward_blocks_range(
                hidden_states, 
                encoder_hidden_states, 
                emb, 
                0, 
                self.cache_start
            )
            # infer [cache_start, cache_end)
            cache_end = np.minimum(self.cache_start + self.num_cache_layer, num_blocks)
            hidden_states_before_cache = hidden_states.clone()
            encoder_hidden_states_before_cache = encoder_hidden_states.clone()
            if t_idx % self.cache_interval == (self.cache_start_steps % self.cache_interval):
                hidden_states, encoder_hidden_states = self._forward_blocks_range(
                    hidden_states, 
                    encoder_hidden_states, 
                    emb, 
                    self.cache_start, 
                    cache_end
                )
                self.delta_cache = hidden_states - hidden_states_before_cache
                self.delta_encoder_cache = encoder_hidden_states - encoder_hidden_states_before_cache
            else:
                hidden_states = hidden_states_before_cache + self.delta_cache
                encoder_hidden_states = encoder_hidden_states_before_cache + self.delta_encoder_cache
            # infer [cache_end, num_blocks)
            hidden_states, encoder_hidden_states = self._forward_blocks_range(
                hidden_states, 
                encoder_hidden_states, 
                emb, 
                cache_end, 
                num_blocks
            )

        return hidden_states, encoder_hidden_states

    def load_weights(self, state_dict, shard=False):
        with torch.no_grad():
            if not shard:
                self.load_state_dict(state_dict)
                return {}
            else:
                weights = state_dict

                for i in range(self.num_layers):
                    if i != 26:
                        q_weight = weights.pop(f"transformer_blocks.{i}.attn1.to_q.weight", None)
                        q_bias = weights.pop(f"transformer_blocks.{i}.attn1.to_q.bias", None)
                        k_weight = weights.pop(f"transformer_blocks.{i}.attn1.to_k.weight", None)
                        k_bias = weights.pop(f"transformer_blocks.{i}.attn1.to_k.bias", None)
                        v_weight = weights.pop(f"transformer_blocks.{i}.attn1.to_v.weight", None)
                        v_bias = weights.pop(f"transformer_blocks.{i}.attn1.to_v.bias", None)

                        # query, key, value的weight和bias权重存在同一个文件中，不会分开存储。
                        if q_weight is not None and k_weight is not None and v_weight is not None:
                            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).transpose(0, 1).contiguous()
                            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0).contiguous()
                            weights[f"transformer_blocks.{i}.attn1.to_qkv.weight"] = qkv_weight
                            weights[f"transformer_blocks.{i}.attn1.to_qkv.bias"] = qkv_bias
                    else:
                        if self.q_weight_cache is None:
                            self.q_weight_cache = weights.pop(f"transformer_blocks.{i}.attn1.to_q.weight", None)
                        if self.q_bias_cache is None:
                            self.q_bias_cache = weights.pop(f"transformer_blocks.{i}.attn1.to_q.bias", None)
                        if self.k_weight_cache is None:
                            self.k_weight_cache = weights.pop(f"transformer_blocks.{i}.attn1.to_k.weight", None)
                        if self.k_bias_cache is None:
                            self.k_bias_cache = weights.pop(f"transformer_blocks.{i}.attn1.to_k.bias", None)
                        if self.v_weight_cache is None:
                            self.v_weight_cache = weights.pop(f"transformer_blocks.{i}.attn1.to_v.weight", None)
                        if self.v_bias_cache is None:
                            self.v_bias_cache = weights.pop(f"transformer_blocks.{i}.attn1.to_v.bias", None)

                qk_weight_cache = self.q_weight_cache is not None and self.k_weight_cache is not None
                if qk_weight_cache and self.v_weight_cache is not None:
                    qkv_weight = torch.cat(
                        [self.q_weight_cache, self.k_weight_cache, self.v_weight_cache], 
                        dim=0
                    ).transpose(0, 1).contiguous()
                    qkv_bias = torch.cat([self.q_bias_cache, self.k_bias_cache, self.v_bias_cache], dim=0).contiguous()
                    weights[f"transformer_blocks.26.attn1.to_qkv.weight"] = qkv_weight
                    weights[f"transformer_blocks.26.attn1.to_qkv.bias"] = qkv_bias

                self.load_state_dict(weights, strict=False, assign=True)
                return weights.keys()
