#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import argparse
import numpy as np

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL infer args")
    parser.add_argument("--tokenizer_path",
                        type=str,
                        required=True,
                        help="Tokenizer path")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        required=True,
                        help="Pretrained model name or path")
    parser.add_argument("--generation_config_path",
                        type=str,
                        required=True,
                        help="Generation config path")
    parser.add_argument("--output_path",
                        type=str,
                        default="./output",
                        help="Output path")
    parser.add_argument("--device_map",
                        type=str,
                        required=True,
                        choices=["cuda", "npu", "cpu", "auto"],
                        help="The device to conduct inference")
    parser.add_argument("--bf16",
                        action="store_true",
                        help="Use bf16")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="Random seed")
    return parser.parse_args()

def set_seed(seed, mode=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    set_seed(args.seed, True)

    # Define toeknizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = (AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                device_map=args.device_map,
                                                trust_remote_code=True, bf16=args.bf16))
    model.generation_config = GenerationConfig.from_pretrained(args.generation_config_path, trust_remote_code=True)

    # 1st dialogue
    query = tokenizer.from_list_format(
        [
            {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
            # Either a local path or an url
            {'text': '这是什么?'},
        ]
    )

    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)

    # 2nd dialogue turn
    response, history = model.chat(tokenizer, '框出图中击掌的位置', history=history)
    print(response)

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save('infer_demo.jpg')
    else:
        print("no box")
