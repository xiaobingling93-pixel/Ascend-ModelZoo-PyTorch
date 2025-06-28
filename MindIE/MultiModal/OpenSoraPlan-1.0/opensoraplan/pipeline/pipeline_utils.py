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
import logging
import importlib
from dataclasses import dataclass

import torch
from torch import Tensor
from tqdm import tqdm
from diffusers.schedulers import PNDMScheduler

from opensoraplan.utils.utils import path_check

from mindiesd import ConfigMixin

PIPELINE_CONFIG_NAME = "model_index.json"
VAE = 'vae'
TEXT_ENCODER = 'text_encoder'
TOKENIZER = 'tokenizer'
TRANSFORMER = 'transformer'
SCHEDULER = 'scheduler'

IMAGE_SIZE = 'image_size'
ENABLE_SEQUENCE_PARALLELISM = 'enable_sequence_parallelism'
FPS = 'fps'
DTYPE = 'dtype'
SET_PATCH_PARALLEL = 'set_patch_parallel'
FROM_PRETRAINED = 'from_pretrained'
NUM_SAMPLING_STEPS = 'num_sampling_steps'
REFINE_SERVER_IP = 'refine_server_ip'
REFINE_SERVER_PORT = 'refine_server_port'
MODEL_TYPE = 'model_type'
logger = logging.getLogger(__name__)  # init python log

OPEN_SORA_PLAN_DEFAULT_IMAGE_SIZE = 512
OPEN_SORA_PLAN_DEFAULT_VIDEO_LENGTH = 17
OPEN_SORA_PLAN_DEFAULT_CACHE_DIR = "cache_dir"
OPEN_SORA_PLAN_DEFAULT_VAE_STRIDE = 8
OPEN_SORA_PLAN_DEFAULT_SCHEDULER = PNDMScheduler
OPEN_SORA_PLAN_DEFAULT_DTYPE = torch.float16

CACHE_DIR = "cache_dir"
VAE_STRIDE = "vae_stride"
VIDEO_LENGTH = "video_length"


class OpenSoraPlanPipelineBase(ConfigMixin):
    config_name = PIPELINE_CONFIG_NAME

    def __init__(self):
        super().__init__()
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        initializers = {
            TEXT_ENCODER: init_text_encoder_plan,
            VAE: init_vae_plan,
            TOKENIZER: init_default_plan
        }

        image_size = kwargs.pop(IMAGE_SIZE, OPEN_SORA_PLAN_DEFAULT_IMAGE_SIZE)
        dtype = kwargs.pop(DTYPE, OPEN_SORA_PLAN_DEFAULT_DTYPE)
        cache_dir = kwargs.pop(CACHE_DIR, OPEN_SORA_PLAN_DEFAULT_CACHE_DIR)
        vae_stride = kwargs.pop(VAE_STRIDE, OPEN_SORA_PLAN_DEFAULT_VAE_STRIDE)
        if vae_stride != 8:
            raise ValueError("Unsupported vae_stride.")
        scheduler = kwargs.pop(SCHEDULER, OPEN_SORA_PLAN_DEFAULT_SCHEDULER)

        real_path = path_check(model_path)
        init_dict, config_dict = cls.load_config(real_path, **kwargs)

        init_list = [VAE, TEXT_ENCODER, TOKENIZER, TRANSFORMER, SCHEDULER]
        pipe_init_dict = {}
        model_init_dict = {}

        all_parameters = inspect.signature(cls.__init__).parameters

        required_param = {k: v for k, v in all_parameters.items() if v.default == inspect.Parameter.empty}
        expected_modules = set(required_param.keys()) - {"self"}
        # init the module from kwargs
        passed_module = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        pipe_init_dict[IMAGE_SIZE] = image_size
        model_init_dict[IMAGE_SIZE] = image_size
        model_init_dict[DTYPE] = dtype
        model_init_dict[CACHE_DIR] = cache_dir
        model_init_dict[VAE_STRIDE] = vae_stride

        for key in tqdm(init_list, desc="Loading open-sora-plan-pipeline compenents"):
            if key not in init_dict:
                raise ValueError(f"Get {key} from init config failed!")
            if key in passed_module:
                pipe_init_dict[key] = passed_module.pop(key)
            else:
                modules, cls_name = init_dict[key]
                if modules == "mindiesd":
                    library = importlib.import_module("opensoraplan")
                else:
                    library = importlib.import_module(modules)
                class_obj = getattr(library, cls_name)
                sub_folder = os.path.join(real_path, key)

                if key == TRANSFORMER:
                    if pipe_init_dict.get(VAE) is None:
                        raise ValueError("Cannot get module 'vae' in init list!")

                    if pipe_init_dict.get(TEXT_ENCODER) is None:
                        raise ValueError("Cannot get module 'text_encoder' in init list!")

                    pipe_init_dict[key] = class_obj.from_pretrained(sub_folder, cache_dir=model_init_dict[CACHE_DIR],
                                                                    torch_dtype=model_init_dict[DTYPE], **kwargs)
                elif key == SCHEDULER:
                    pipe_init_dict[key] = scheduler
                else:
                    initializer = initializers.get(key, init_default_plan)
                    pipe_init_dict[key] = initializer(class_obj, sub_folder, model_init_dict, kwargs)

        if pipe_init_dict.get(TRANSFORMER) is None:
            raise ValueError("Cannot get module 'transformer' in init list!")
        video_length = pipe_init_dict.get(TRANSFORMER).config.video_length
        pipe_init_dict[VIDEO_LENGTH] = video_length

        return cls(**pipe_init_dict)

    def progress_bar(self, iterable=None, total=None):
        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")


def init_text_encoder_plan(class_obj, sub_folder, model_init_dict, kwargs):
    return class_obj.from_pretrained(sub_folder, cache_dir=model_init_dict[CACHE_DIR],
                                     torch_dtype=model_init_dict[DTYPE])


def init_vae_plan(class_obj, sub_folder, model_init_dict, kwargs):
    height = width = model_init_dict[IMAGE_SIZE] // model_init_dict[VAE_STRIDE]
    latent_size = (height, width)
    vae = class_obj.from_pretrained(sub_folder, latent_size, cache_dir=model_init_dict[CACHE_DIR],
                                    **kwargs).to(dtype=model_init_dict[DTYPE])
    return vae


def init_default_plan(class_obj, sub_folder, model_init_dict, kwargs):
    return class_obj.from_pretrained(sub_folder, **kwargs)