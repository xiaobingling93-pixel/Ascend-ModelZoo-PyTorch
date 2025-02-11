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
# limitations under the License


import os

import torch
import torch.nn as nn
from mindiesd import ConfigMixin
from .model_load_utils import load_state_dict


DIFFUSER_SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"


class DiffusionModel(nn.Module):
    config_class = ConfigMixin
    weigths_name = DIFFUSER_SAFETENSORS_WEIGHTS_NAME

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        dtype = kwargs.pop('dtype', None) # get dtype from kwargs
        if not (dtype in {torch.bfloat16, torch.float16}):
            raise ValueError("dtype should be a torch.bfloat16 or torch.float16")

        # 1. check model_path and weights_path
        real_path = os.path.abspath(model_path)
        if not (os.path.exists(real_path) and os.path.isdir(real_path)):
            raise ValueError(f"{real_path} is invalid!")

        if not issubclass(cls.config_class, ConfigMixin):
            raise ValueError("config_class is not subclass of ConfigMixin.")
        
        if cls.weigths_name is None:
            raise ValueError("weigths_name is not defined.")

        weights_path = os.path.join(real_path, cls.weigths_name)
        if not (os.path.exists(weights_path) and os.path.isfile(weights_path)):
            raise ValueError(f"'{cls.weigths_name}' is not found in '{model_path}'!")

        # 2. load config_class from json
        init_dict, _ = cls.config_class.load_config(real_path, **kwargs)
        config = cls.config_class(**init_dict)

        # 3. init model with config
        model = cls(config)

        # 4. load model weights
        state_dict = load_state_dict(weights_path)
        model._load_weights(state_dict)

        # 5. model to dtype
        if dtype is not None:
            model.to(dtype)
        return model
    
    def _load_weights(self, state_dict):
        with torch.no_grad():
            self.load_state_dict(state_dict)