import torch
import torch_npu
import torch.nn as nn
from einops import rearrange

try:
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except ImportError:
    xFuserLongContextAttention = None
    
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def attn_processor(self, attn_type):
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        else:
            raise Exception('Not supported attention type...')

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.logical_not(attn_mask.bool()).to(q.device)
        else:
            attn_mask = attn_mask.bool().to(q.device)

        x = torch_npu.npu_fusion_attention(q, k, v, q.size(2), "BSND", atten_mask=attn_mask, scale=q.size(-1) ** (-0.5), keep_prob=1.0, sparse_mode=0)[0]

        return x        

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        assert xFuserLongContextAttention is not None; # 'to use sequence parallel attention, xFuserLongContextAttention should be imported...'
        hybrid_seq_parallel_attn = xFuserLongContextAttention()
        x = hybrid_seq_parallel_attn(
            None, q, k, v, causal=causal
        )
        return x

