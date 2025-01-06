#!/usr/bin/env python
# coding=utf-8
# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import safetensors.torch


SAFETENSORS_EXTENSION = "safetensors"
EMA_STATE_DICT = "ema_state_dict"
STATE_DICT = "state_dict"
CPU = "cpu"


def load_state_dict_sd(model_path):
    name = os.path.basename(model_path).split('.')[-1] # get weights name
    if name.endswith("ckpt"):
        weight = torch.load(model_path, map_location=CPU)
        if (EMA_STATE_DICT in weight):
            weight = weight[EMA_STATE_DICT]
            weight = {key.replace("module.", ""): value for key, value in weight.items()}
        elif STATE_DICT in weight:
            weight = weight[STATE_DICT]
        return weight
    elif name == SAFETENSORS_EXTENSION: # diffuser model use same name
        return safetensors.torch.load_file(model_path, device=CPU) # first load on cpu
    else:
        # to support hf shard model weights
        return torch.load(model_path, map_location=CPU) # first load on cpu