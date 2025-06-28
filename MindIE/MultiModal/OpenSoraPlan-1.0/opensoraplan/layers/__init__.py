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

from .attention import (
    AttnBlock3D,
    AttnBlock,
    LinAttnBlock,
    LinearAttention,
)
from .conv import CausalConv3d, Conv2d
from .resnet_block import ResnetBlock2D, ResnetBlock3D
from .ops import nonlinearity, normalize
from .updownsample import (
    SpatialDownsample2x,
    SpatialUpsample2x,
    TimeDownsample2x,
    TimeUpsample2x,
    TimeDownsampleRes2x,
    TimeUpsampleRes2x,
)