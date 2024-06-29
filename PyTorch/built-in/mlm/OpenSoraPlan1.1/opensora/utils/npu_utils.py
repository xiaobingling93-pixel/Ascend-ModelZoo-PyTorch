# Copyright 2024 Huawei Technologies Co., Ltd
from typing import List, Optional
import importlib
from functools import lru_cache

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer


@lru_cache
def is_npu_available():
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu

    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


if is_npu_available():
    import torch_npu


class NpuRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initialize NPU RMSNorm normalization layer
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x.to(self.weight.dtype), self.weight, epsilon=self.eps)[0]


def replace_module(model, submodule_key, module):
    """Replace all the submodule of the model with module that contains submodule_key"""
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


