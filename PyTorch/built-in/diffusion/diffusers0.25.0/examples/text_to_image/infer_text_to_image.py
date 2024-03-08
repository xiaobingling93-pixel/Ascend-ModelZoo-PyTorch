#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import os
import torch
import argparse
from diffusers import StableDiffusionXLPipeline

# Adapter to NPU
import torch_npu
from torch_npu.contrib import transfer_to_npu

parser = argparse.ArgumentParser(description="Simple example of a infer script.")
parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)
parser.add_argument("--ckpt_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="ckpt from finetine output.")
parser.add_argument("--output_path", type=str, default="./infer", help="infer output path.")
parser.add_argument("--device_id", type=int, default=0, help="device id to infer.")

args = parser.parse_args()

torch.npu.set_device(args.device_id)
os.makedirs(args.output_path, exist_ok=True)

if args.mixed_precision == "fp16":
    torch_dtype = torch.float16
elif args.mixed_precision == "bf16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

# init model pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(args.ckpt_path, torch_dtype=torch_dtype).to("npu")


# set prompt
prompts = list()
prompts.append("cute dragon creature.")

generator = torch.Generator(device="cpu").manual_seed(0)

for prompt in prompts:
    image = pipe(prompt=prompt, generator=generator).images
    image[0].save(f"{args.output_path}/{prompt}.png")

