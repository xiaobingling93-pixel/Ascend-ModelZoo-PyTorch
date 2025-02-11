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
import torch_npu


def weight_switch(weights, prefix_key, new_weight, old_weight, transpose=None):
    if prefix_key + old_weight in weights:
        weights[prefix_key + new_weight] = weights.pop(prefix_key + old_weight)
        if transpose:
            weights[prefix_key + new_weight] = weights[prefix_key + new_weight
                                                       ].transpose(*transpose).contiguous()


def get_attn_weight(weights, prefix_key, cross_attention, fuse=True):
    cache_weights = {}
    # If self attention, fuse the qkv, crcoss attention fuse the kv
    qkv = ["q", "k", "v"] if not cross_attention else ["k", "v"]
    for wb in ["weight", "bias"]:
        if fuse:
            weight_name = [prefix_key + f'to_{i}.' + wb for i in qkv]
            conds = [w in weights for w in weight_name]
            # If weights do not contain all the q,k,v, put them in the cache_weights
            # And the cache_weights will be added in the next shard weights
            if not all(conds) and any(conds):
                for w in weight_name:
                    if w in weights:
                        cache_weights[w] = weights.pop(w)
            # weights contain all the q k v weight
            if all(conds):
                qkv_weight = []
                for w in weight_name:
                    qkv_weight.append(weights.pop(w))
                mid_key = "".join(qkv) + "_" if cross_attention else ""
                if wb == "weight":
                    weights[prefix_key + "qkv_proj." + mid_key + wb] = torch.cat(
                        qkv_weight, dim=0).transpose(-1, 0).contiguous()
                else:
                    weights[prefix_key + "qkv_proj." + mid_key + wb] = torch.cat(qkv_weight, dim=0)
        else:
            for q in qkv:
                weight_switch(weights, prefix_key, q + '_proj.' + wb, f'to_{q}.' + wb)

        if cross_attention:
            weight_switch(weights, prefix_key, 'qkv_proj.q_' + wb, 'to_q.' + wb,
                          transpose=(-1, 0) if wb == 'weight' else None)

        # switch out linear
        weight_switch(weights, prefix_key, 'out_proj.' + wb, 'to_out.0.' + wb)
    return cache_weights