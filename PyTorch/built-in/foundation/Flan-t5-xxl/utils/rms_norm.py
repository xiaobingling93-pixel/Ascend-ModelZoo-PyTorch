import torch
import torch_npu

def forward(self, hidden_states):
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]