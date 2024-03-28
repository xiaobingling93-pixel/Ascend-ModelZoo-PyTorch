# Copyright 2023 Huawei Technologies Co., Ltd
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
import torch_npu
import torch.nn.functional as F
from sat.model.base_model import BaseModel
from sat.model.mixins import BaseMixin
from sat.model.official.vit_model import ViTProperty, ImagePatchEmbeddingMixin, InterpolatedPositionEmbeddingMixin, gelu
from sat import mpu


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=1., attention_dropout=0.):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, n, attention_mask, pse):

        if self.causal:
            output = torch_npu.npu_fusion_attention(
                q, k, v, n, "BSND",# SBH
                pse=pse,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=self.softmax_scale,
                pre_tockens=65536,  # seq_len
                next_tockens=65536,  # 0
                keep_prob=1 - self.dropout_p,
            )[0]
            return output


class IdentityMixin(BaseMixin):
    def __init__(self):
        super().__init__()

    def final_forward(self, logits, **kwargs):
        return logits[:, 1:]


def memory_efficient_attention_min(query, key, value, dropout):
    """
    xformers.ops.memory_efficient_attention 小算子实现
    """
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn = query @ key.transpose(-2, -1)
    attn = attn.softmax(-1)
    attn = F.dropout(attn, dropout)
    attn = attn @ value
    return attn.transpose(1, 2)


class XAttn(BaseMixin):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.core_attention_flash = FlashSelfAttention(causal=True, softmax_scale=self.scale, attention_dropout=0.)

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        dropout_p = 0. # xformers does not support dropout for eva hidden size

        query_layer = query_layer.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)
        out = self.core_attention_flash(query_layer, key_layer, value_layer, query_layer.shape[2], None, None)
        return out
    
    def attention_forward(self, hidden_states, mask, **kw_args):
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)

        B, N, C = hidden_states.shape
        mixed_raw_layer = mixed_raw_layer.reshape(B, N, 3, self.num_attention_heads_per_partition, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
        query_layer, key_layer, value_layer = mixed_raw_layer[0], mixed_raw_layer[1], mixed_raw_layer[2]

        dropout_fn = self.attention_dropout if self.training else None

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.contiguous().view(B, N, -1)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output


class NewLayerForward(BaseMixin):
    def __init__(self):
        super().__init__()

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        self = self.transformer.layers[kw_args['layer_id']]
        
        attention_input = hidden_states

        # Self attention.
        attention_output = self.input_layernorm(self.attention(attention_input, mask, **kw_args))

        # DropPath for attention
        if self.training and self.drop_path > 0.:
            if mpu.get_cuda_rng_tracker is not None:
                # drop_path must use model parallel rng tracker
                # the tracker is initialized as seed of `seed + model_parallel_rank`
                # deepspeed act-ckpt record the model parallel tracker states
                with mpu.get_cuda_rng_tracker().fork():
                    # drop_path percentage 0, others 1/(1-p)
                    random_tensor = (1-self.drop_path
                                    + torch.rand((attention_output.shape[0],), dtype=attention_output.dtype, device=attention_output.device)).floor_() / (1-self.drop_path)
                    attention_output = random_tensor.view(-1, 1, 1) * attention_output
        
        # Residual connection.
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states

        # MLP.
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input, **kw_args))

        # DropPath for mlp
        if self.training and self.drop_path > 0.:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    random_tensor = (1-self.drop_path
                                    + torch.rand((mlp_output.shape[0],), dtype=mlp_output.dtype, device=mlp_output.device)).floor_() / (1-self.drop_path)
                    mlp_output = random_tensor.view(-1, 1, 1) * mlp_output

        # Second residual connection.
        output = mlp_input + mlp_output

        return output

class EVA2CLIPModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        args.max_sequence_length = property.pre_len + property.num_patches + property.post_len
        if 'activation_func' not in kwargs:
            kwargs['activation_func'] = gelu
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kwargs)
        self.transformer.property = property
        self.add_mixin("patch_embedding", ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, property))
        self.add_mixin("pos_embedding", InterpolatedPositionEmbeddingMixin())
        self.add_mixin("final", IdentityMixin())
        self.add_mixin("newpost", NewLayerForward())
        self.add_mixin("xattn", XAttn(args.hidden_size // args.num_attention_heads))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EVA2CLIP', 'EVA2CLIP Configurations')
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1) # [cls] by default
        group.add_argument('--post-len', type=int, default=0) # empty by default, but sometimes with special tokens, such as [det] in yolos.
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--patch-size', type=int, default=16)
        return parser
