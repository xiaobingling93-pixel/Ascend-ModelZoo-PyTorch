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


import torch.nn as nn
from mindiesd import get_activation_layer, Linear


class Mlp(nn.Module):

    def __init__(self,
                 features_in,
                 features_hidden=None,
                 features_out=None,
                 act_layer="gelu",
                 norm_layer=None,
                 bias=True,
                 op_type=None):
        super().__init__()

        features_out = features_out or features_in
        features_hidden = features_hidden or features_in

        if op_type is None:
            self.fc1 = nn.Linear(features_in, features_hidden, bias=bias)
            self.fc2 = nn.Linear(features_hidden, features_out, bias=bias)
        else:
            self.fc1 = Linear(features_in, features_hidden, bias=bias, op_type=op_type)
            self.fc2 = Linear(features_hidden, features_out, bias=bias, op_type=op_type)

        self.act = act_layer() if not isinstance(act_layer, str) else get_activation_layer(act_layer)
        self.norm = norm_layer(features_hidden) if norm_layer is not None else nn.Identity()
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x
