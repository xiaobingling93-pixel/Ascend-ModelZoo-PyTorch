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


import torch.nn as nn
from ..utils import is_npu_available


def compile_pipe(pipe, cfg=None):
    if is_npu_available():
        device = 'npu'
        if hasattr(pipe, "text_encoder") and isinstance(pipe.text_encoder, nn.Module):
            pipe.text_encoder.to(device)
        if hasattr(pipe, "transformer") and isinstance(pipe.transformer, nn.Module):
            pipe.transformer.to(device)
        if hasattr(pipe, "vae") and isinstance(pipe.vae, nn.Module):
            pipe.vae.to(device)
        return pipe
    else:
        raise RuntimeError("NPU is not available.")