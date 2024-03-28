# Copyright 2023 Huawei Technologies Co., Ltd
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

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))
except ModuleNotFoundError:
    from sat.helpers import print_rank0
    print_rank0('Please install apex to use fused_layer_norm, fall back to torch.nn.LayerNorm', level='DEBUG')
    import torch
    class LayerNorm(torch.nn.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def forward(self, x):
            # if cpu and float16, calculate in float32 for both x and weight, bias.
            if str(x.device) == 'cpu' and x.dtype in[torch.float16, torch.bfloat16]:
                return torch.nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float()).to(x.dtype)
            else:
                return super().forward(x)    

import torch
import torch_npu
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        return torch_npu.npu_rms_norm(hidden_states.to(torch.float32), self.weight.to(torch.float32), epsilon=self.variance_epsilon)[0].to(input_dtype)
