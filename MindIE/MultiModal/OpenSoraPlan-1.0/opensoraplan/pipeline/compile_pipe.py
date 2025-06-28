#!/usr/bin/env python
# coding=utf-8
# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

from ..utils import is_npu_available
from ..acceleration.dit_cache_common import DiTCacheManager

CFG_MAX_STEP = 10000


def compile_pipe(pipe, cache_manager: DiTCacheManager = None,
                 cfg_last_step: int = CFG_MAX_STEP):
    if not isinstance(cfg_last_step, int):
        raise TypeError(f"Expected int for cfg_last_step, but got {type(cfg_last_step).__name__}")

    if is_npu_available():
        device = 'npu'
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder.to(device)
        else:
            raise TypeError("Please input valid pipeline")
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(device)
        if hasattr(pipe, "vae"):
            pipe.vae.to(device)

        if cache_manager is not None:
            if not hasattr(cache_manager, "use_cache"):
                raise TypeError("Please input valid cache_manager")
            pipe.transformer.cache_manager = cache_manager
        if cfg_last_step != CFG_MAX_STEP:
            pipe.cfg_last_step = cfg_last_step
        return pipe
    else:
        raise RuntimeError("NPU is not available.")
