# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
from typing import List, Optional
import importlib
from functools import lru_cache

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer


AUTOCAST_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

@lru_cache
def is_npu_available():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec('torch_npu') is None:
        return False

    import torch
    import torch_npu

    try:
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


if is_npu_available():
    import torch_npu


def seed_all(is_gpu=True, seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    if is_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    else:
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)


class FlashAttention(torch.nn.Module):
    def __init__(self, attention_dropout=0):
        super().__init__()
        self.attention_dropout = attention_dropout

    def forward(self, query, key, value):
        heads = query.shape[2]
        attention_mask = None
        output = torch_npu.npu_fusion_attention(
            query, key, value, heads, input_layout='BSND',
            pse=None,
            atten_mask=attention_mask,
            scale=1.0 / math.sqrt(query.shape[-1]),
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1. - self.attention_dropout,
            sync=False,
            inner_precise=0,
        )[0]
        return output
