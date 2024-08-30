import os
import torch
import torch.nn as nn
import torch_npu
import transformers
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NpuLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).to(torch.get_default_dtype()), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        _, seq_len = position_ids.shape

        if seq_len != 1:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(seq_len=seq_len, device=x.device)
            return (
                self.cos_cached[:, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :seq_len, ...].to(dtype=x.dtype),
            )
        else:
            # x: [bs, num_attention_heads, seq_len, head_size]
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            # Force float32 since bfloat16 loses precision on long contexts
            # See https://github.com/huggingface/transformers/pull/29285
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


unsqueeze_dim = 2 if os.getenv('use_flash_attention_2') == 'true' else 1


def apply_fused_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return torch_npu.npu_rotary_mul(q, cos, sin), torch_npu.npu_rotary_mul(k, cos, sin)


def replace_with_torch_npu_llama_rope():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = NpuLlamaRotaryEmbedding
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_fused_rotary_pos_emb
