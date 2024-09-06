import torch
import torch.nn as nn
import torch_npu
import transformers


class NpuLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        return torch_npu.npu_rms_norm(
            hidden_states,
            self.weight.to(torch.float32),
            epsilon=self.variance_epsilon
        )[0].to(input_dtype)


def replace_with_torch_npu_llama_rmsnorm():
    transformers.models.llama.modeling_llama.LlamaRMSNorm = NpuLlamaRMSNorm
