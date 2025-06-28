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

import torch
from einops import rearrange


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = func(self, x, *args, **kwargs)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x
    return wrapper


def nonlinearity(x):
    return x * torch.sigmoid(x)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )