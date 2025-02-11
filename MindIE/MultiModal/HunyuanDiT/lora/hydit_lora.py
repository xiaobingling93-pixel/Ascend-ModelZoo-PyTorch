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
from safetensors import safe_open


def multi_lora(args, pipeline):
    transformer_state_dict = pipeline.transformer.state_dict()
    lora_state_dict = {}
    with safe_open(args.lora_ckpt, framework="pt", device=args.device) as f:
        for k in f.keys():
            lora_state_dict[k[17:]] = f.get_tensor(k)

    num_blocks = pipeline.transformer.config.depth
    merge_state_dict = load_lora(transformer_state_dict, lora_state_dict, num_blocks, lora_scale=1.0)
    return merge_state_dict


def load_lora(transformer_state_dict, lora_state_dict, num_blocks, lora_scale):

    for i in range(num_blocks):
        Wqkv = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_B.weight"],
                            lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn1.qkv_proj.weight"] += lora_scale * Wqkv

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_B.weight"],
                                lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn1.out_proj.weight"] += lora_scale * out_proj

        q_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_B.weight"],
                              lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.q_proj.weight"] += lora_scale * q_proj

        kv_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_B.weight"],
                               lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.kv_proj.weight"] += lora_scale * kv_proj

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_B.weight"],
                                lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn2.out_proj.weight"] += lora_scale * out_proj
    
    q_proj = torch.matmul(lora_state_dict["pooler.q_proj.lora_B.weight"],
                          lora_state_dict["pooler.q_proj.lora_A.weight"])
    transformer_state_dict["pooler.attn.q_proj.weight"] += lora_scale * q_proj
    
    return transformer_state_dict
