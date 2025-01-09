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


import os
import random
import argparse
import time
from pathlib import Path
import logging

import torch

from hydit import HunyuanDiTPipeline, compile_pipe, set_seeds_generator
from lora import multi_lora

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="ckpts/hydit", help="Path to the model directory")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--device", type=str, default="npu", help="NPU")
    parser.add_argument("--prompt", type=str, default="渔舟唱晚", help="The prompt for generating images")
    parser.add_argument("--prompt_list", type=str, default="prompts/example_prompts.txt", help="The prompt list")
    parser.add_argument("--test_acc", action="store_true", help="Run or not 'example_prompts.txt'")
    parser.add_argument("--input_size", type=int, nargs='+', default=[1024, 1024], help='Image size (h, w)')
    parser.add_argument("--type", type=str, default="fp16", help="The torch type is fp16 or bf16")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-NPU batch size")
    parser.add_argument('--seed', type=int, default=42, help="A seed for all the prompts")
    parser.add_argument("--infer_steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA checkpoint")
    parser.add_argument("--lora_ckpt", type=str, default="ckpts/lora", help="LoRA checkpoint")
    return parser.parse_args()


def get_dtype(args):
    dtype = torch.bfloat16
    if args.type == 'bf16':
        dtype = torch.bfloat16
    elif args.type == 'fp16':
        dtype = torch.float16
    else:
        logger.error("Not supported.")
    return dtype


def get_seed(args):
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 1_000_000)
    if not isinstance(seed, int):
        raise ValueError(f"The type of seed must be int, but got {type(seed)}.")
    if seed < 0:
        raise ValueError(f"Input seed must be a non-negative integer, but got {seed}.")
    return set_seeds_generator(seed, device=args.device)


def get_prompts(args):
    if not args.test_acc:
        prompts = args.prompt
        prompts = [prompts.strip()]
    else:
        lines_list = []
        prompt_list_path = os.path.join(args.path, args.prompt_list)
        with open(prompt_list_path, 'r') as file:
            for line in file:
                line = line.strip()
                lines_list.append(line)
        prompts = lines_list
    return prompts


def infer(args):
    if not Path(args.path).exists():
        raise ValueError(f"args.path not exists: {Path(args.path)}")
    if len(args.input_size) != 2:
        raise ValueError(f"The length of args.input_size must be 2, but got {len(args.input_size)}")
    input_size = (args.input_size[0], args.input_size[1])

    torch.npu.set_device(args.device_id)
    dtype = get_dtype(args)
    seed_generator = get_seed(args)

    pipeline = HunyuanDiTPipeline.from_pretrained(model_path=args.path, input_size=input_size, dtype=dtype)
    pipeline = compile_pipe(pipeline)

    if args.use_lora:
        merge_state_dict = multi_lora(args, pipeline)
        pipeline.transformer.load_state_dict(merge_state_dict)

    prompts = get_prompts(args)
    loops = len(prompts)

    save_dir = Path('results')
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True)
    
    now_time = time.localtime(time.time())
    time_dir_name = time.strftime("%m%d%H%M%S", now_time)
    time_dir = save_dir / Path(time_dir_name)
    time_dir.mkdir(exist_ok=True)

    pipeline_total_time = 0.0
    for i in range(loops):
        prompt = prompts[i]

        start_time = time.time()

        result_images = pipeline(
            prompt=prompt,
            num_images_per_prompt=args.batch_size,
            num_inference_steps=args.infer_steps,
            seed_generator=seed_generator,
        )[0]

        pipeline_time = time.time() - start_time
        logger.info("HunyuanDiT No.{%d} time: %.3f", i, pipeline_time)
        torch.npu.empty_cache()

        if i >= 2:
            pipeline_total_time += pipeline_time
        
        save_path = time_dir / f"{i}.png"
        result_images[0].save(save_path)
        torch.npu.empty_cache()
    
    if args.test_acc:
        if loops <= 2:
            raise ValueError(f"The loops must be larger than 2 but got {loops}")
        pipeline_average_time = pipeline_total_time / (loops - 2)
        logger.info("HunyuanDiT pipeline_average_time: %.3f", pipeline_average_time)
    torch.npu.empty_cache()


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)