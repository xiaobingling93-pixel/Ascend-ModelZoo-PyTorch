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

import argparse
import logging
import time
import os
import csv
import json

import torch

from cogview3plus import CogView3PlusPipeline, set_random_seed, CogView3PlusTransformer2DModel
from cogview3plus.utils.file_utils import standardize_path
from mindiesd import CacheAgent, CacheConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            batch_size: int,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)
        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(prompt_file, max_num_prompts)
        else:
            print("This operation is not supported!")

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration

        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret

    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))

    def load_prompts_hpsv2(self, file_path: str, max_num_prompts: int):
        with open(file_path, 'r') as file:
            all_prompts = json.load(file)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                count += 1
                if max_num_prompts and count >= max_num_prompts:
                    break

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an image using the CogView3-Plus-3B model.")

    # Define arguments for prompt, model path, etc.
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts/example_prompts.txt",
        help="A text file of prompts for generating images.",
    )
    parser.add_argument(
        "--prompt_file_type",
        choices=["plain", "parti", "hpsv2"],
        default="plain",
        help="Type of prompt file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result images.",
    )
    parser.add_argument(
        "--info_file_save_path",
        type=str,
        default="./image_info.json",
        help="Path to save image information file.",
    )
    parser.add_argument(
        "--model_path", type=str, default="/data/CogView3B", help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.0, help="The guidance scale for classifier-free guidance."
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--max_num_prompts",
        default=0,
        type=int,
        help="Limit the number of prompts (0: no limit).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps for inference.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16 or fp16")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument('--cache_algorithm', type=str, default="None", help="The type of optimization algorithm")

    return parser.parse_args()


def infer(args):
    torch.npu.set_device(args.device_id)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        set_random_seed(args.seed)

    # Load the pre-trained model with the specified precision
    args.model_path = standardize_path(args.model_path)
    pipe = CogView3PlusPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    transformer = CogView3PlusTransformer2DModel.from_pretrained(os.path.join(args.model_path, 'transformer'), torch_dtype=dtype)
    pipe.transformer = transformer
    pipe = pipe.to("npu")

    # attention cache
    if args.cache_algorithm == "attention":
        steps_count = args.num_inference_steps
        blocks_count = pipe.transformer.config.num_layers
        config = CacheConfig(
            method="attention_cache",
            blocks_count=blocks_count,
            steps_count=steps_count,
            step_start=15,
            step_end=37,
            step_interval=2
            )
        agent = CacheAgent(config)
        pipe.transformer.use_cache = True
        for block in pipe.transformer.transformer_blocks:
            block.cache = agent

    use_time = 0
    prompt_loader = PromptLoader(args.prompt_file,
                                 args.prompt_file_type,
                                 args.batch_size,
                                 args.num_images_per_prompt,
                                 args.max_num_prompts)

    infer_num = 0
    image_info = []
    current_prompt = None
    for i, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']

        print(f"[{infer_num + n_prompts}/{len(prompt_loader)}]: {prompts}")
        infer_num += args.batch_size

        start_time = time.time()
        images = pipe(
            prompt=prompts,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            image_size=(args.height, args.width),
        )

        if i > 1: # do not count the time spent inferring the first 0 to 1 batches
            use_time += time.time() - start_time

        for j in range(n_prompts):
            image_save_path = os.path.join(args.save_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)

    infer_num = infer_num - 2 * args.batch_size # do not count the time spent inferring the first 2 batches.
    print(f"[info] infer number: {infer_num}; use time: {use_time:.3f}s\n"
          f"average time: {use_time / infer_num:.3f}s\n")

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)

    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)
