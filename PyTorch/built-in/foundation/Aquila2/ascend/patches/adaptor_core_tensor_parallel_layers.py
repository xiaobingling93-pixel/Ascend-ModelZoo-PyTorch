# Copyright 2024 Huawei Technologies Co., Ltd
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

import torch.nn.functional as F
import megatron
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region


def VocabParallelEmbeddingForward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
    # Get the embeddings.
    output_parallel = F.embedding(masked_input, self.weight,
                                  self.padding_idx, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq,
                                  self.sparse)
    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output


megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = VocabParallelEmbeddingForward
