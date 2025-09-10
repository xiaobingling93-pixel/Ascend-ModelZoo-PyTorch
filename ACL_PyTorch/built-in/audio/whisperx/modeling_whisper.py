# Copyright 2025 Huawei Technologies Co., Ltd
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

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

import whisper
from whisper.model import Linear, LayerNorm, Whisper
from whisper.decoding import PyTorchInference



class MyMultiHeadSelfAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int, n_ctx: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.kv = Linear(in_features=self.key.weight.shape[0], out_features=self.key.weight.shape[1] + self.value.weight.shape[1])
        self.n_ctx = n_ctx

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            kv_cache: Optional[dict] = None,
            updated_kv_positions: Optional[torch.LongTensor] = None,
            actual_seq_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None
    ):
        q = self.query(x)

        n_batch, n_ctx, n_state = q.shape
        max_sample_len = self.n_ctx
        # decoder - self_attention
        k_key = "key"
        v_key = "value"
        # Prefill
        if k_key not in kv_cache:
            kv_cache[k_key] = torch.zeros(n_batch, max_sample_len, n_state, dtype=x.dtype, device=x.device)
            kv_cache[v_key] = torch.zeros(n_batch, max_sample_len, n_state, dtype=x.dtype, device=x.device)
            kv = self.kv(x)
            k, v = kv.chunk(2, dim=-1)
            kv_cache[k_key][:, :n_ctx, :] = k.detach().contiguous()
            kv_cache[v_key][:, :n_ctx, :] = v.detach().contiguous()
        # Decode
        else:
            new_kv = self.kv(x[:, -1:])
            new_k, new_v = new_kv.chunk(2, dim=-1)
            tmp_ids = updated_kv_positions.expand(n_batch)
            torch_npu.scatter_update_(kv_cache[k_key], tmp_ids, new_k, 1)
            torch_npu.scatter_update_(kv_cache[v_key], tmp_ids, new_v, 1)

            k = kv_cache[k_key]
            v = kv_cache[v_key]

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        D = n_state // self.n_head
        # Prefill用PFA
        if n_ctx > 1:
            mask = mask.to(torch.bool) if mask is not None and n_ctx > 1 else None
            sparse_mode = 1 if mask is not None and n_ctx > 1 else 0
            attn = torch_npu.npu_prompt_flash_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                num_heads=self.n_head,
                input_layout="BNSD",
                scale_value=1 / math.sqrt(D),
                atten_mask=mask[:n_ctx, :n_ctx] if mask is not None else None,
                sparse_mode=sparse_mode
            )
        # Decode用IFA
        else:
            attn = torch_npu.npu_incre_flash_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                num_heads=self.n_head,
                input_layout="BNSD",
                scale_value=1 / math.sqrt(D),
                atten_mask=None,
                actual_seq_lengths=actual_seq_len,
                kv_padding_size=kv_padding_size
            )

        w_v = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(w_v)


class MyMultiHeadCrossAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.kv = Linear(in_features=self.key.weight.shape[0],
                         out_features=self.key.weight.shape[1] + self.value.weight.shape[1])

    def forward(
            self,
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        # decoder - cross_attention
        k_key = "key"
        v_key = "value"
        if k_key in kv_cache:
            k = kv_cache[k_key]
            v = kv_cache[v_key]
        else:
            kv = self.kv(xa)
            k, v = kv.chunk(2, dim=-1)
            kv_cache[k_key] = k.contiguous()
            kv_cache[v_key] = v.contiguous()

        n_batch, n_ctx, n_state = q.shape
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        mask = mask.to(torch.bool) if mask is not None and n_ctx > 1 else None
        sparse_mode = 1 if mask is not None and n_ctx > 1 else 0
        D = n_state // self.n_head
        attn = torch_npu.npu_prompt_flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            num_heads=self.n_head,
            input_layout="BNSD",
            scale_value=1 / math.sqrt(D),
            atten_mask=mask[:n_ctx, :n_ctx] if mask is not None else None,
            sparse_mode=sparse_mode
        )

        w_v = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(w_v)


class MyResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_ctx: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MyMultiHeadSelfAttention(n_state, n_head, n_ctx)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MyMultiHeadCrossAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
            self,
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            kv_cache: Optional[dict] = None,
            updated_kv_positions: Optional[torch.LongTensor] = None,
            actual_seq_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None
    ):
        x = x + self.attn(self.attn_ln(x),
                          mask=mask,
                          kv_cache=kv_cache['attn'],
                          actual_seq_len=actual_seq_len,
                          kv_padding_size=kv_padding_size,
                          updated_kv_positions=updated_kv_positions)
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache['cross_attn'])
        x = x + self.mlp(self.mlp_ln(x))
        return x


class MyTextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                MyResidualAttentionBlock(n_state, n_head, n_ctx, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
            self,
            x: Tensor,
            xa: Tensor,
            positional_embedding: Tensor = None,
            kv_cache: Optional[dict] = None,
            updated_kv_positions: Optional[torch.LongTensor] = None,
            actual_seq_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None
    ):
        pass


class PrefillTextDecoder(MyTextDecoder):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)

    def forward(
            self,
            x: Tensor,
            xa: Tensor,
            positional_embedding: Tensor = None,
            kv_cache: Optional[dict] = None,
            updated_kv_positions: Optional[torch.LongTensor] = None,
            actual_seq_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None
    ):
        offset = 0
        x = (
                self.token_embedding(x)
                + self.positional_embedding[offset: offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for layer_index, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache[layer_index],
                      updated_kv_positions=updated_kv_positions)

        x = self.ln(x)
        logits = (
                x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class DecodeTextDecoder(MyTextDecoder):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)

    def forward(
            self,
            x: Tensor,
            xa: Tensor,
            positional_embedding: Tensor,
            kv_cache: Optional[dict] = None,
            updated_kv_positions: Optional[torch.LongTensor] = None,
            actual_seq_len: Optional[list] = None,
            kv_padding_size: Optional[torch.LongTensor] = None
    ):
        x = (self.token_embedding(x) + positional_embedding)
        x = x.to(xa.dtype)

        for layer_index, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache[layer_index], actual_seq_len=actual_seq_len,
                      kv_padding_size=kv_padding_size,
                      updated_kv_positions=updated_kv_positions)

        x = self.ln(x)
        logits = (
                x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


def rewrite_multi_head_attention_forward(model):
    wk = model.key.weight
    wv = model.value.weight
    model.kv = Linear(in_features=wk.shape[0], out_features=wk.shape[1] + wv.shape[1])
    model.kv.weight = nn.Parameter(torch.concat([wk, wv], dim=0), requires_grad=False)
    wk_bias = model.key.bias if model.key.bias is not None else torch.zeros(wk.shape[0])
    wv_bias = model.value.bias if model.value.bias is not None else torch.zeros(wv.shape[0])
    model.kv.bias = nn.Parameter(torch.concat([wk_bias, wv_bias], dim=0), requires_grad=False)

    def forward(
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            kv_cache: Optional[dict] = None,
            actual_seq_len: Optional[list] = None,
    ):
        q = model.query(x)

        # encoder
        if kv_cache is None:
            kv = model.kv(x)
            k, v = kv.chunk(2, dim=-1)

        # decoder - cross_attention
        if kv_cache is not None and xa is not None:
            k_key = "key"
            v_key = "value"
            if k_key in kv_cache:
                k = kv_cache[k_key]
                v = kv_cache[v_key]
            else:
                kv = model.kv(xa)
                k, v = kv.chunk(2, dim=-1)
                kv_cache[k_key] = k.contiguous()
                kv_cache[v_key] = v.contiguous()

        # decoder - self_attention
        if kv_cache is not None and xa is None:
            k_key = "key"
            v_key = "value"
            if k_key in kv_cache:
                k = kv_cache[k_key]
                v = kv_cache[v_key]
                new_kv = model.kv(x[:, -1:])
                new_k = new_kv[..., :wk.shape[0]]
                new_v = new_kv[..., wk.shape[0]:]
                kv_cache[k_key] = torch.cat([k.contiguous(), new_k.contiguous()], dim=1).detach()
                kv_cache[v_key] = torch.cat([v.contiguous(), new_v.contiguous()], dim=1).detach()
                k, v = kv_cache[k_key], kv_cache[v_key]
            else:
                kv = model.kv(x)
                k, v = kv.chunk(2, dim=-1)
                kv_cache[k_key] = k.contiguous()
                kv_cache[v_key] = v.contiguous()

        n_batch, n_ctx, n_state = q.shape
        q = q.view(*q.shape[:2], model.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], model.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], model.n_head, -1).permute(0, 2, 1, 3)

        mask = mask.to(torch.bool) if mask is not None and n_ctx > 1 else None
        sparse_mode = 1 if mask is not None and n_ctx > 1 else 0
        D = n_state // model.n_head

        at = torch_npu.npu_prompt_flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            num_heads=model.n_head,
            input_layout="BNSD",
            scale_value=1 / math.sqrt(D),
            atten_mask=mask[:n_ctx, :n_ctx] if mask is not None else None,
            sparse_mode=sparse_mode
        )

        qk = None
        w_v = at.permute(0, 2, 1, 3).flatten(start_dim=2)
        return model.out(w_v), qk

    model.forward = forward


def modify_model(model, options, device):
    print("modify model...")

    # rewrite attention forward
    for block in model.encoder.blocks:
        rewrite_multi_head_attention_forward(block.attn)
    for block in model.decoder.blocks:
        rewrite_multi_head_attention_forward(block.attn)
        rewrite_multi_head_attention_forward(block.cross_attn)
    original_decoder = model.decoder

    # split the original decoder into prefill stage and decode stage
    prefill_decoder = PrefillTextDecoder(
        model.dims.n_vocab,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
        model.dims.n_text_head,
        model.dims.n_text_layer
    )
    prefill_decoder.load_state_dict(original_decoder.state_dict())

    decode_decoder = DecodeTextDecoder(
        model.dims.n_vocab,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
        model.dims.n_text_head,
        model.dims.n_text_layer
    )
    decode_decoder.load_state_dict(original_decoder.state_dict())

    model.prefill_decoder = prefill_decoder
    model.decode_decoder = decode_decoder

    if options.fp16:
        model = model.half()
        for module in model.modules():
            # 在Whisper源码中，LayerNorm层需要接收fp32数据，因此需要特殊处理
            if isinstance(module, nn.LayerNorm):
                module = module.float()

    return model.eval().to(device)


def rewrite_inference_logits(n_layer):
    def _patched_logits(self, tokens, audio_features) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
            self.kv_cache = [
                {
                    'attn': {},
                    'cross_attn': {}
                }
                for _ in range(n_layer)
            ]
            return self.model.prefill_decoder(tokens, audio_features, kv_cache=self.kv_cache)

        actual_seq_len = tokens.shape[-1]
        updated_kv_positions = torch.tensor([actual_seq_len - 1], dtype=torch.long, device=tokens.device)
        kv_padding_size = torch.tensor([448 - actual_seq_len], dtype=torch.long, device=tokens.device)

        offset = actual_seq_len - 1
        positional_embedding = self.model.decode_decoder.positional_embedding[offset: offset + 1]
        tokens = tokens[:, -1:].contiguous().clone()

        torch._dynamo.mark_static(tokens)
        torch._dynamo.mark_static(audio_features)
        torch._dynamo.mark_static(positional_embedding)
        for i in range(n_layer):
            torch._dynamo.mark_static(self.kv_cache[i]['attn']["key"])
            torch._dynamo.mark_static(self.kv_cache[i]['attn']["value"])
            torch._dynamo.mark_static(self.kv_cache[i]['cross_attn']["key"])
            torch._dynamo.mark_static(self.kv_cache[i]['cross_attn']["value"])
        torch._dynamo.mark_static(kv_padding_size)

        return self.model.decode_decoder(tokens, audio_features, positional_embedding, self.kv_cache,
                                         actual_seq_len=[actual_seq_len], kv_padding_size=kv_padding_size,
                                         updated_kv_positions=updated_kv_positions)

    PyTorchInference.logits = _patched_logits


def rewrite_whisper_logits(n_layer):
    # for language detection only
    def _whisper_logits(self, tokens, audio_features) -> Tensor:
        kv_cache = [
            {
                    'attn': {},
                    'cross_attn': {}
            }
            for _ in range(n_layer)
        ]
        return self.prefill_decoder(tokens, audio_features, kv_cache=kv_cache)

    Whisper.logits = _whisper_logits


def get_whisper_model(whisper_model_path, whisper_decode_options, device):
    model = whisper.load_model(whisper_model_path)
    model = modify_model(model, whisper_decode_options, device)
    rewrite_inference_logits(model.dims.n_text_layer)
    rewrite_whisper_logits(model.dims.n_text_layer)

    return model