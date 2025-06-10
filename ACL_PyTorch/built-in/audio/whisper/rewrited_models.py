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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch_npu

from whisper.model import Linear, LayerNorm, MultiHeadAttention, ResidualAttentionBlock


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
        # Prefill用FPA
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
                          updated_kv_positions=updated_kv_positions)[0]
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache['cross_attn'])[0]
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
