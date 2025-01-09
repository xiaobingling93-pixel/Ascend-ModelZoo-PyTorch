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


import os
import inspect
import importlib

import torch
from tqdm import tqdm

from mindiesd import ConfigMixin


PIPELINE_CONFIG_NAME = "model_index.json"
SCHEDULER = 'scheduler'
TEXT_ENCODER = 'text_encoder'
TEXT_ENCODER_2 = 'text_encoder_2'
TOKENIZER = 'tokenizer'
TOKENIZER_2 = 'tokenizer_2'
TRANSFORMER = 'transformer'
VAE = 'vae'

HUNYUAN_DEFAULT_IMAGE_SIZE = (1024, 1024)
HUNYUAN_INPUT_SIZE = "input_size"
HUNYUAN_DEFAULT_DTYPE = torch.float16
HUNYUAN_DTYPE = "dtype"


class HunYuanPipeline(ConfigMixin):
    config_name = PIPELINE_CONFIG_NAME

    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        input_size = kwargs.pop(HUNYUAN_INPUT_SIZE, HUNYUAN_DEFAULT_IMAGE_SIZE)
        dtype = kwargs.pop(HUNYUAN_DTYPE, HUNYUAN_DEFAULT_DTYPE)
        if model_path is None:
            raise ValueError("The model_path should not be None.")
        init_dict, _ = cls.load_config(model_path, **kwargs)

        init_list = [SCHEDULER, TEXT_ENCODER, TEXT_ENCODER_2, TOKENIZER, TOKENIZER_2, TRANSFORMER, VAE]
        pipe_init_dict = {}

        all_parameters = inspect.signature(cls.__init__).parameters

        required_param = {k: v for k, v in all_parameters.items() if v.default == inspect.Parameter.empty}
        expected_modules = set(required_param.keys()) - {"self"}
        # init the module from kwargs
        passed_module = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}

        for key in tqdm(init_list, desc="Loading hunyuan-dit-pipeline components"):
            if key not in init_dict:
                raise ValueError(f"Get {key} from init config failed!")
            if key in passed_module:
                pipe_init_dict[key] = passed_module.pop(key)
            else:
                modules, cls_name = init_dict[key]
                if modules == "mindiesd":
                    library = importlib.import_module("hydit")
                else:
                    library = importlib.import_module(modules)
                class_obj = getattr(library, cls_name)

                sub_folder = os.path.join(model_path, key)

                if key == TRANSFORMER:
                    pipe_init_dict[key] = class_obj.from_pretrained(
                        sub_folder, input_size=input_size, dtype=dtype, **kwargs)
                elif key == VAE:
                    pipe_init_dict[key] = class_obj.from_pretrained(sub_folder, dtype=dtype, **kwargs)
                elif key == SCHEDULER:
                    pipe_init_dict[key] = class_obj.from_config(sub_folder)
                else:
                    pipe_init_dict[key] = class_obj.from_pretrained(sub_folder, **kwargs)

        pipe_init_dict[HUNYUAN_INPUT_SIZE] = input_size
        pipe_init_dict[HUNYUAN_DTYPE] = dtype

        return cls(**pipe_init_dict)