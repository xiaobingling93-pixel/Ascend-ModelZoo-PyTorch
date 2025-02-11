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


def rearrange_flatten_t(x):
    x_shape = x.shape
    x = x.transpose(1, 2)
    return x.view((x_shape[0] * x_shape[2]), x_shape[1], x_shape[3], x_shape[4])


def rearrange_unflatten_t(x, b):
    x_shape = x.shape
    x = x.view(b, x_shape[0] // b, x_shape[1], x_shape[2], x_shape[3])
    return x.transpose(1, 2)