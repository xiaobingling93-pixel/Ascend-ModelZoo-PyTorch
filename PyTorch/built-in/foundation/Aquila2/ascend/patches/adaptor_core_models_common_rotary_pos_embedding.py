# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import sys

import torch_npu
import megatron
from megatron.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb

def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    return torch_npu.npu_rotary_mul(t, freqs.cos(), freqs.sin())

megatron.core.models.common.rotary_pos_embedding.apply_rotary_pos_emb = apply_rotary_pos_emb

for k, v in sys.modules.items():
    if 'megatron' in k and hasattr(v, 'apply_rotary_pos_emb'):
        setattr(v, 'apply_rotary_pos_emb', apply_rotary_pos_emb)
