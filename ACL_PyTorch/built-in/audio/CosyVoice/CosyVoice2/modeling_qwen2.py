# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
""" PyTorch Qwen2 model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    is_flash_attn_2_available,
    logging
)
from .configuration_qwen2 import Qwen2Config


logger = logging.get_logger(__name__)


QWEN2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen2-7B-beta",
]


# Ascend优化：Add/Norm昇腾自定义融合算子
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self,
                hidden_states,
                residual: Optional[torch.Tensor] = None):
        if residual is None:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0], hidden_states
        else:
            y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
        return y, x


# Ascend优化：提前计算位置编码，无需在每层layer中重复计算
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x=None, seq_len=None):
        if x is None and seq_len is None:
            return self.cos_cached, self.sin_cached

        return (
            self.cos_cached.to(dtype=x.dtype),
            self.sin_cached.to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


# Ascend优化：PFA/IFA自定义算子替换，kv cache固定shape并在指定位置更新
class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # 优化Attention部分逻辑，替换torch_npu算子
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        rotary_emb_cos: Optional[torch.Tensor] = None,
        rotary_emb_sin: Optional[torch.Tensor] = None,        
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # 利用已经提前计算好的位置编码数据对q,k值进行更新
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        rotary_emb_cos.to(value_states.dtype),
                                                        rotary_emb_sin.to(value_states.dtype))

        if use_cache and past_key_value is not None:
            # 把计算好的kv值更新到kv cahce中
            tmp_ids = updated_kv_positions.reshape(-1)
            torch_npu.scatter_update_(past_key_value.key_cache[self.layer_idx], tmp_ids, key_states, 1)
            torch_npu.scatter_update_(past_key_value.value_cache[self.layer_idx], tmp_ids, value_states, 1)
            kv_states = past_key_value[self.layer_idx] if q_len == 1 else (key_states, value_states)
            key_states = kv_states[0]
            value_states = kv_states[1]


        if q_len > 1:
            # prefill阶段利用PFA自定义算子执行计算，因为bs为1，mask固定为下三角全为0上三角全为负无穷的倒三角mask矩阵
            attn_output = torch_npu.npu_prompt_flash_attention(query_states, key_states.contiguous(),
                                                               value_states.contiguous(), num_heads=self.num_heads,
                                                               input_layout="BSND",
                                                               scale_value=1 / math.sqrt(self.head_dim),
                                                               pre_tokens=65535, next_tokens=0,
                                                               atten_mask=attention_mask,
                                                               num_key_value_heads=self.num_key_value_heads)
        else:
            # decode阶段利用IFA自定义算子执行计算，qkv的sequence都为1，该算子采用tiling下沉，视为静态算子，支持整图下发 
            attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states.contiguous(),
                                                              value_states.contiguous(), num_heads=self.num_heads,
                                                              input_layout="BSND",
                                                              scale_value=1 / math.sqrt(self.head_dim),
                                                              atten_mask=None,
                                                              actual_seq_lengths=actual_seq_len,
                                                              kv_padding_size=kv_padding_size,
                                                              num_key_value_heads=self.num_key_value_heads)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "sdpa": Qwen2SdpaAttention,
}


# Ascend优化：每层layer的前后rms替换为昇腾自定义算子
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        rotary_emb_cos: Optional[torch.Tensor] = None,
        rotary_emb_sin: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # rms计算替换为昇腾自定义融合算子
        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            updated_kv_positions=updated_kv_positions,
            kv_padding_size=kv_padding_size,
            actual_seq_len=actual_seq_len,
            rotary_emb_cos=rotary_emb_cos,
            rotary_emb_sin=rotary_emb_sin,
            use_cache=use_cache,
        )

        # rms计算替换为昇腾自定义融合算子
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (residual, hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Ascend优化：forward函数利用torchair编译为图模式，利用cache接口避免重复编译
@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_theta = config.rope_theta

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        # torchair编译参数，编译Qwen2Model的forward部分
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        # tiling下沉，主要针对IFA算子，使其算子tiling操作在AICPU上执行
        config.experimental_config.tiling_schedule_optimize = True
        # torchair的cache编译，保证模型编译cache文件，避免重复推理
        self.cached_decode = tng.inference.cache_compile(self.decode, config=config)
        self.cached_prefill = tng.inference.cache_compile(self.prefill, config=config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_rotary_cos_sin(self, position_ids):
        cos, sin = self.rotary_emb()
        f_position_ids = position_ids.flatten()
        cos = torch.index_select(cos, 0, f_position_ids)
        sin = torch.index_select(sin, 0, f_position_ids)
        cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
        sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
        return cos, sin

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lm_head: Optional[object] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # prefill和decode需要编译为两个不同的模型
        if inputs_embeds.size(1) > 1:
            return self.cached_prefill(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                updated_kv_positions,
                kv_padding_size,
                actual_seq_len,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                lm_head        
            )
        else:
            return self.cached_decode(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                updated_kv_positions,
                kv_padding_size,
                actual_seq_len,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                lm_head
            )

    def decode(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lm_head: Optional[object] = None       
    ):
        return self._forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            updated_kv_positions,
            kv_padding_size,
            actual_seq_len,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            lm_head             
        )

    def prefill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lm_head: Optional[object] = None      
    ):
        return self._forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            updated_kv_positions,
            kv_padding_size,
            actual_seq_len,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            lm_head             
        )


    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        updated_kv_positions: Optional[torch.LongTensor] = None,
        kv_padding_size: Optional[torch.LongTensor] = None,
        actual_seq_len: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lm_head: Optional[object] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")


        # prefill阶段初始化kv cache，decode阶段对kv cache进行更新
        # 固定kv cache为最大shape，避免内存的重复申请和拷贝，也保证了模型的静态shape，可整图下发推理
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                kv_shape = (
                    batch_size, self.config.max_position_embeddings,
                    self.config.num_key_value_heads,
                    self.config.hidden_size // self.config.num_attention_heads)
                past_key_values = ()
                for _ in range(self.config.num_hidden_layers):
                    k_cache = torch.zeros(kv_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                    v_cache = torch.zeros(kv_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                    past_key_values += ((k_cache, v_cache),)
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        past_key_values_length = self.max_position_embeddings if seq_length == 1 else 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # 此处统一计算位置编码，在每个layer中取对应位置的值
        rotary_emb_cos, rotary_emb_sin = self._prepare_decoder_rotary_cos_sin(position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        residual = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 执行layer层推理
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_residual=residual,
                position_ids=position_ids,
                past_key_value=past_key_values,
                updated_kv_positions=updated_kv_positions,
                kv_padding_size=kv_padding_size,
                actual_seq_len=actual_seq_len,
                rotary_emb_cos=rotary_emb_cos,
                rotary_emb_sin=rotary_emb_sin,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            residual = layer_outputs[0]
            hidden_states = layer_outputs[1]

            if use_cache:
                next_decoder_cache = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        # norm计算，此处替换为昇腾融合算子
        hidden_states, _ = self.norm(hidden_states, residual)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        hidden_states = out[0]
        # 由于logits最后也只取[:,-1,:]，相当于只取最新seq位置上的数据，l
        # 所以在全量的最后线性层计算可以只对最新的seq位置做计算，降低计算量
        bs, seq, hidden = hidden_states.size()
        if seq > 1:
            gather_index = torch.ones(bs, dtype=torch.int64, device=hidden_states.device) * (seq - 1)
            gather_index = gather_index.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 1, hidden)
            hidden_states = torch.gather(hidden_states, 1, gather_index)
        logits = lm_head(hidden_states)
        logits = logits.float()
        return out, logits


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        prompt_length: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        对CosyVoice2模型中使用的Qwen模型进行昇腾适配优化，具体优化点有：
        1. 固定KV CACHE大小，避免重复申请内存和拷贝
        2. 替换部分算子为昇腾自定义算子
        3. 首层计算位置编码避免重复计算
        4. 在decode阶段，固定输入shape大小，保证整图下发

        模型有以下输入：
        1. attention_mask
        2. inputs_embeds：CosyVoice会把inputs_ids处理embeding后输入模型
        3. past_key_values：kv cache，在每次推理后会进行更新
        4. position_ids：位置id，在每次推理后会进行更新
        5. prompt_length：实际输入长度，在prefill阶段为首token长度，后续每次推理长度加1
        """

        # 每次推理前对输入数据进行昇腾适配处理，处理为昇腾自定义算子所需类型参数
        updated_kv_positions, past_key_values, position_ids, kv_padding_size, actual_seq_len = self.prepare_data(inputs_embeds, past_key_values, prompt_length)

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "kv_padding_size": kv_padding_size,
            "actual_seq_len": actual_seq_len,
            "attention_mask": attention_mask,
        }

        # prefill阶段由于输出token长度不固定，为动态shape推理。decode阶段把输入固定为静态，保证整图静态推理。
        if inputs_embeds.shape[1] == 1:
            self._mark_model_inputs_static(model_inputs)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 主要推理阶段，利用torchair编译为整图推理
        outputs, logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            updated_kv_positions=updated_kv_positions,
            kv_padding_size=kv_padding_size,
            actual_seq_len=actual_seq_len,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lm_head=self.lm_head
        )

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )      

    # Ascend优化：把数据输入处理为Ascend优化所需要的格式和类型
    def prepare_data(self, inputs_embeds, past_key_values, prompt_length):
        bsz = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        if past_key_values:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if seq_length > 1:
            updated_kv_positions = torch.zeros(bsz, dtype=torch.long, device=inputs_embeds.device)
            position_ids = None
        else:
            updated_kv_positions = torch.ones(bsz, dtype=torch.long, device=inputs_embeds.device) * (prompt_length - 1)
            position_ids = torch.tensor([prompt_length], device=inputs_embeds.device)

        # ifa Computational optimization inputs
        kv_padding_size = torch.tensor(self.config.max_position_embeddings - prompt_length, device=inputs_embeds.device)
        actual_seq_len = ([prompt_length])

        return updated_kv_positions, past_key_values, position_ids, kv_padding_size, actual_seq_len
    
    # Ascend优化：固定input shape，使能静态推理，模型整图下发
    def _mark_model_inputs_static(self, model_inputs):
        for key, value in model_inputs.items():
            if key == "past_key_values" and value is not None:
                for i in range(self.config.num_hidden_layers):
                    torch._dynamo.mark_static(value[i][0])
                    torch._dynamo.mark_static(value[i][1])
            elif isinstance(value, torch.Tensor):
                torch._dynamo.mark_static(value)    

