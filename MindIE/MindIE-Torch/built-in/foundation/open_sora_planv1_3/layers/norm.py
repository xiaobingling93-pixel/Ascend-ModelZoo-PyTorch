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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_normalization_helper(norm_type: str, norm_dim: int, eps: float = 1e-5):
    match norm_type:
        case None:
            return nn.Identity()
        case 'layer_norm':
            return nn.LayerNorm(norm_dim, eps=eps)
        case _:
            raise ValueError(f"Unsupported norm_type:{norm_type}.")


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x, shift, scale):
        return F.layer_norm(x, normalized_shape=[self.hidden_size], weight=scale, bias=shift, eps=self.eps)


def Normalize(in_channels, num_groups=32, norm_type="groupnorm"):
    if norm_type == "groupnorm":
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "layernorm":
        return VideoLayerNorm(num_channels=in_channels, eps=1e-6)
    

class VideoLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps, elementwise_affine=True)
    def forward(self, x):
        if x.dim() == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = self.norm(x)
            x = rearrange(x, "b t h w c -> b c t h w")
        else:
            x = rearrange(x, "b c h w -> b h w c")
            x = self.norm(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x
