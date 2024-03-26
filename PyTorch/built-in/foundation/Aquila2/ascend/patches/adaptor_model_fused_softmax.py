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

import torch_npu
import megatron


def is_kernel_available(self, mask, b, np, sq, sk):
    return (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 32 < sk <= 2048  # sk must be 32 ~ 2048
            and sq % 16 == 0  # sq must be divisor of 16
            and sk % 16 == 0  # sk must be divisor of 16
    )


def forward_fused_softmax(self, input, mask):
    return torch_npu.npu_scaled_masked_softmax(input, mask, self.scale, False)


megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available
megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax
