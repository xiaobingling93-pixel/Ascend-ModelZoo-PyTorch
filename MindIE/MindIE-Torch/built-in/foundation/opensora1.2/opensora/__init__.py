#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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

from .vae import (VideoAutoencoder, VideoAutoencoderConfig)
from .pipeline import (OpenSoraPipeline12, compile_pipe)
from .schedulers import RFlowScheduler
from .stdit3 import (STDiT3Config, STDiT3)
from .utils import (set_random_seed, append_score_to_prompts, extract_prompts_loop, merge_prompt, prepare_multi_resolution_info, 
                    split_prompt, is_npu_available, exists, default, Patchify, Depatchify)
from .layer import (approx_gelu, CaptionEmbedder, PatchEmbed3D, PositionEmbedding2D, SizeEmbedder, TimestepEmbedder, RotaryEmbedding,
                    Mlp, AdaLayerNorm, PatchGroupNorm3d, GroupNorm3dAdapter, all_to_all_with_pad, get_spatial_pad, get_temporal_pad,
                    set_spatial_pad, set_temporal_pad, split_sequence, gather_sequence, set_parallel_manager, get_sequence_parallel_group,
                    get_sequence_parallel_size, rearrange_flatten_t, rearrange_unflatten_t, Conv3dAdapter, PatchConv3d, Attention, MultiHeadCrossAttention)