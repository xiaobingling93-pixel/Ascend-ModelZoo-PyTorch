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
import importlib
from einops import rearrange


def rearrange_flatten_t(x):
    x_shape = x.shape
    x = x.transpose(1, 2)
    return x.view((x_shape[0] * x_shape[2]), x_shape[1], x_shape[3], x_shape[4])


def rearrange_unflatten_t(x, b):
    x_shape = x.shape
    x = x.view(b, x_shape[0] // b, x_shape[1], x_shape[2], x_shape[3])
    return x.transpose(1, 2)


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = func(self, x, *args, **kwargs)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x
    return wrapper


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) or isinstance(t, list) else ((t,) * length)


MODULES_BASE = "open_sora_planv1_3.layers."


def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)