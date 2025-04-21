# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
import math
import torch
import torch_npu
import numpy as np
from stepvideo.parallel import (
        get_llm_tensor_model_parallel_rank,
        get_llm_tensor_model_parallel_world_size
)

DTYPE_FP16_MIN = float(np.finfo(np.float16).min)


def _get_alibi_slopes(n_heads):
    # get slopes
    n = 2**math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
    m0 = torch.tensor(2.0**(-8.0 / n), dtype=torch.float32).to("cpu")
    slopes = torch.pow(m0, torch.arange(1, n + 1, dtype=torch.float32).to("cpu"))
    if n < n_heads:
        m1 = torch.tensor(2.0**(-4.0 / n), dtype=torch.float32).to("cpu")
        mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2, dtype=torch.float32).to("cpu"))
        slopes = torch.cat([slopes, mm])
    return slopes


def _get_mask(seq_len, b, n):
    slopes = _get_alibi_slopes(n)
    tril = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool)).to(torch.int32)
    bias_row = torch.arange(seq_len).view(1, -1)
    bias_cols = torch.arange(seq_len).view(-1, 1)
    bias = -torch.sqrt(bias_cols - bias_row)
    bias = bias.view(1, seq_len, seq_len) * slopes.view(-1, 1, 1)
    bias = bias.masked_fill(tril == 0, DTYPE_FP16_MIN)
    return bias


class FlashSelfAttention(torch.nn.Module):
    def __init__(
        self,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, cu_seqlens=None, max_seq_len=None):
        if cu_seqlens is None:
            tp_rank = get_llm_tensor_model_parallel_rank()
            tp_size = get_llm_tensor_model_parallel_world_size()
            alibi_mask = _get_mask(q.size(1), q.size(0), q.size(2) * tp_size)
            alibi_mask = alibi_mask[:, (tp_rank * q.size(2)):(tp_rank * q.size(2) + q.size(2)), :, :].to(q.dtype).to(q.device)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=alibi_mask)
            output = output.transpose(1, 2) 
        else:
            raise ValueError('cu_seqlens is not supported!')

        return output
    
