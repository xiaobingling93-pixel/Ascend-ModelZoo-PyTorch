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

import collections.abc
from itertools import repeat

import torch.nn as nn
from .activation import get_activation_fn


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            features_in,
            act_layer,
            features_hidden=None,
            features_out=None,
            norm_layer=None,
            bias=True,
    ):
        super().__init__()
        features_out = features_out or features_in
        features_hidden = features_hidden or features_in
        to_2tuple = self._ntuple(2)
        bias = to_2tuple(bias)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(features_in, features_hidden, bias=bias[0])  
        self.act = get_activation_fn(act_layer) 
        self.norm = norm_layer(features_hidden) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(features_hidden, features_out, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

    def _ntuple(self, n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
                return tuple(x)
            return tuple(repeat(x, n))
        return parse