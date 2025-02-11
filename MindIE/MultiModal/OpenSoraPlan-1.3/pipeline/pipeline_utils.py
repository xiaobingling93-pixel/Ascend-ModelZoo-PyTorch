#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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
import inspect
import importlib
from typing import List, Optional, Union

from tqdm import tqdm
import torch
from mindiesd import ConfigMixin


PIPELINE_CONFIG_NAME = "model_index.json"
VAE = 'vae'
TEXT_ENCODER = 'text_encoder'
TOKENIZER = 'tokenizer'
TRANSFORMER = 'transformer'
SCHEDULER = 'scheduler'


class DiffusionPipeline(ConfigMixin):
    config_name = PIPELINE_CONFIG_NAME

    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        dtype = kwargs.pop('dtype', None)
        real_path = os.path.abspath(model_path)
        if not (os.path.exists(real_path) and os.path.isdir(real_path)):
            raise ValueError("model path is invalid!")

        init_dict, config_dict = cls.load_config(real_path, **kwargs)

        all_parameters = inspect.signature(cls.__init__).parameters
        required_param = {k: v for k, v in all_parameters.items() if v.default is inspect.Parameter.empty}

        # init the module from kwargs
        passed_module = {k: kwargs.pop(k) for k in required_param if k in kwargs}
        from_diffusers = None if '_diffusers_version' not in config_dict else config_dict['_diffusers_version']
        for key, item in tqdm(init_dict.items(), desc="Loading pipeline components..."):
            if key in passed_module:
                init_dict[key] = passed_module.pop(key)
            else:
                modules, cls_name = item
                if from_diffusers:
                    try:
                        library = importlib.import_module("mindiesd")
                        class_obj = getattr(library, cls_name)
                    except ImportError:
                        print("Warning:", f"Cannot import {cls_name} from mindiesd. Use diffuser.")
                        library = importlib.import_module(modules)
                        class_obj = getattr(library, cls_name)
                else:
                    library = importlib.import_module(modules)
                    class_obj = getattr(library, cls_name)
                sub_folder = os.path.join(real_path, key)
                if key.startswith(TOKENIZER):
                    init_dict[key] = class_obj.from_pretrained(sub_folder, **kwargs)
                elif key.startswith(SCHEDULER):
                    init_dict[key] = class_obj.from_config(sub_folder, **kwargs)
                else:
                    init_dict[key] = class_obj.from_pretrained(sub_folder, **kwargs).to(dtype)

        return cls(**init_dict)
    

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):


    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed."
                                " Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps