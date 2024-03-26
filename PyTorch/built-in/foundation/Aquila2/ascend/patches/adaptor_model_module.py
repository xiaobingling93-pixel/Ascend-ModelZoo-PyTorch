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

import torch
import megatron
from megatron.model.module import conversion_helper


def fp32_to_float16(val, float16_convertor):
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype in [torch.float16, torch.bfloat16]:
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


megatron.model.module.fp32_to_float16 = fp32_to_float16
megatron.model.module.float16_to_fp32 = float16_to_fp32
