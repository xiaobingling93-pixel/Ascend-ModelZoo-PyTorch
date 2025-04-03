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
import logging
import csv
import json

import torch

from diffusers import AutoencoderKL
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from transformers.modeling_utils import logger as tf_logger

from mindiesd import CacheConfig, CacheAgent
from hydit import HunyuanDiTPipeline, HunyuanDiT2DModel, DDPMScheduler, set_seeds_generator
from hydit.utils import file_utils
from lora import multi_lora

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.num_images_per_prompt = num_images_per_prompt
        self.max_num_prompts = max_num_prompts

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file)
        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file)
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(prompt_file)
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
        }
        for _ in range(self.num_images_per_prompt):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
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

    def load_prompts_plain(self, file_path: str):
        with file_utils.safe_open(file_path, "r", encoding="utf-8",
                                  permission_mode=file_utils.SAFEOPEN_FILE_PERMISSION) as file:
            for i, line in enumerate(file):
                if self.max_num_prompts and i == self.max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str):
        with file_utils.safe_open(file_path, "r", encoding="utf-8",
                                  permission_mode=file_utils.SAFEOPEN_FILE_PERMISSION) as file:
            # Skip the first line
            next(file)
            tsv_file = csv.reader(file, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if self.max_num_prompts and i == self.max_num_prompts:
                    break

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))

    def load_prompts_hpsv2(self, file_path: str):
        with file_utils.safe_open(file_path, "r", encoding="utf-8",
                                  permission_mode=file_utils.SAFEOPEN_FILE_PERMISSION) as file:
            all_prompts = json.load(file)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                count += 1
                if self.max_num_prompts and count >= self.max_num_prompts:
                    break

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="ckpts/t2i", help="Path to the model directory")
    parser.add_argument("--save_result_path", type=str, default="./results", help="Path to save result images")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--device", type=str, default="npu", help="NPU")
    parser.add_argument("--prompt", type=str, default="渔舟唱晚", help="The prompt for generating images")
    parser.add_argument("--test_acc", action="store_true", help="Run or not 'example_prompts.txt'")
    parser.add_argument("--prompt_file", type=str, default="prompts/example_prompts.txt", help="The prompt list")
    parser.add_argument("--prompt_file_type", choices=["plain", "parti", "hpsv2"], default="plain",
                        help="Type of prompt file")
    parser.add_argument("--info_file_save_path", type=str, default="./image_info.json",
                        help="Path to save image information file")
    parser.add_argument("--max_num_prompts", default=0, type=int, help="Limit the number of prompts (0: no limit)")

    parser.add_argument("--input_size", type=int, nargs="+", default=[1024, 1024], help="Image size (h, w)")
    parser.add_argument("--type", type=str, default="fp16", help="The torch type is fp16 or bf16")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-NPU batch size")
    parser.add_argument("--seed", type=int, default=42, help="A seed for all the prompts")
    parser.add_argument("--infer_steps", type=int, default=100, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale for classifier-free")

    parser.add_argument("--use_lora", action="store_true", help="Use LoRA checkpoint")
    parser.add_argument("--lora_ckpt", type=str, default="ckpts/lora", help="LoRA checkpoint")

    parser.add_argument("--use_cache", action="store_true", help="Run or not using cache")
    parser.add_argument("--step_start", type=int, default=9, help="The start iteration steps of cache")
    parser.add_argument("--step_interval", type=int, default=2, help="The step interval of cache")
    parser.add_argument("--block_start", type=int, default=5, help="The block start of cache")
    parser.add_argument("--num_blocks", type=int, default=30, help="The num blocks of cache")

    parser.add_argument("--beta_end", type=float, default=0.02, help="Scheduler beta_end=0.03 if model<=1.1")
    parser.add_argument("--use_style_cond", action="store_true", help="Use style condition. Only for model<=1.1")
    parser.add_argument("--size_cond", type=int, nargs="+", default=None,
                        help="Size condition used in sampling. Default=[1024, 1024]. Only for model<=1.1")

    parser = add_attentioncache_args(parser)
    return parser.parse_args()


def add_attentioncache_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Attention Cache args")
    group.add_argument("--use_attentioncache", action="store_true", help="Run or not using attention cache")
    group.add_argument("--start_step", type=int, default=9, help="The start iteration steps of cache")
    group.add_argument("--attentioncache_interval", type=int, default=6, help="The step interval of cache")
    group.add_argument("--end_step", type=int, default=97, help="The end iteration steps of cache")
    return parser


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


def get_save_path(args):
    save_dir = args.save_result_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    now_time = time.localtime(time.time())
    time_dir_name = time.strftime("%m%d%H%M%S", now_time)
    time_dir = os.path.join(save_dir, time_dir_name)
    os.makedirs(time_dir)
    logger.info(f"Save result image to {time_dir}")
    return time_dir


def get_pipeline(args):
    tf_logger.setLevel('ERROR')
    if len(args.input_size) != 2:
        raise ValueError(f"The length of args.input_size must be 2, but got {len(args.input_size)}")
    input_size = (args.input_size[0], args.input_size[1])
    dtype = get_dtype(args)

    scheduler = DDPMScheduler(beta_end=args.beta_end)

    text_encoder_path = os.path.join(args.path, "clip_text_encoder")
    text_encoder = BertModel.from_pretrained(text_encoder_path).to(args.device)
    tokenizer_path = os.path.join(args.path, "tokenizer")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    mt5_path = os.path.join(args.path, "mt5")
    text_encoder_2 = T5EncoderModel.from_pretrained(mt5_path).to(args.device).eval().to(dtype)
    tokenizer_2 = T5Tokenizer.from_pretrained(mt5_path)

    vae_path = os.path.join(args.path, "sdxl-vae-fp16-fix")
    vae = AutoencoderKL.from_pretrained(vae_path).to(args.device)

    transformer_path = os.path.join(args.path, "model")
    transformer = HunyuanDiT2DModel.from_pretrained(transformer_path,
                                                    input_size=input_size,
                                                    size_cond=args.size_cond,
                                                    use_style_cond=args.use_style_cond,
                                                    dtype=dtype)
    transformer = transformer.to(args.device).eval()

    pipeline = HunyuanDiTPipeline(scheduler=scheduler,
                                  text_encoder=text_encoder,
                                  tokenizer=tokenizer,
                                  text_encoder_2=text_encoder_2,
                                  tokenizer_2=tokenizer_2,
                                  transformer=transformer,
                                  vae=vae,
                                  args=args,
                                  input_size=input_size)
    return pipeline


def infer(args):
    time_path = get_save_path(args)
    seed_generator = get_seed(args)
    pipeline = get_pipeline(args)

    if args.use_lora:
        merge_state_dict = multi_lora(args, pipeline)
        pipeline.transformer.load_state_dict(merge_state_dict)

    if args.use_attentioncache:
        config = CacheConfig(
            method="attention_cache",
            blocks_count=len(pipeline.transformer.blocks),
            steps_count=args.infer_steps,
            step_start=args.start_step,
            step_interval=args.attentioncache_interval,
            step_end=args.end_step
        )
    else:
        config = CacheConfig(
            method="attention_cache",
            blocks_count=len(pipeline.transformer.blocks),
            steps_count=args.infer_steps
        )
    cache_agent = CacheAgent(config)
    for block in pipeline.transformer.blocks:
        block.cache = cache_agent

    pipeline_total_time = 0.0
    infer_num = 0
    image_info = []
    current_prompt = None
    prompt_loader = PromptLoader(args.prompt_file, args.prompt_file_type, args.batch_size, args.max_num_prompts)
    if args.test_acc:
        for i, input_info in enumerate(prompt_loader):
            prompts = input_info['prompts']
            catagories = input_info['catagories']
            save_names = input_info['save_names']

            start_time = time.time()
            result_images = pipeline(
                prompt=prompts[0],
                num_images_per_prompt=args.batch_size,
                num_inference_steps=args.infer_steps,
                seed_generator=seed_generator,
            )[0]
            pipeline_time = time.time() - start_time
            logger.info("HunyuanDiT [%d/%d] time: %.3f", infer_num + 1, len(prompt_loader), pipeline_time)
            torch.npu.empty_cache()

            if infer_num >= (2 * args.batch_size):
                pipeline_total_time += pipeline_time
            infer_num += args.batch_size

            for j, img in enumerate(result_images):
                save_path = os.path.join(time_path, f"{save_names[j]}.png")
                img.save(save_path)
                torch.npu.empty_cache()

                if current_prompt != prompts[j]:
                    current_prompt = prompts[j]
                    image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})
                image_info[-1]['images'].append(save_path)

        if infer_num <= (2 * args.batch_size):
            raise ValueError(f"The number of prompts must be greater than {2*args.batch_size}, but got {infer_num}")
        pipeline_average_time = pipeline_total_time / (infer_num - (2 * args.batch_size))
        logger.info("HunyuanDiT pipeline_average_time: %.3f", pipeline_average_time)

        # Save image information to a json file
        if args.prompt_file_type != "plain":
            if os.path.exists(args.info_file_save_path):
                os.remove(args.info_file_save_path)
            with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as file:
                json.dump(image_info, file)
    else:
        prompts = args.prompt
        prompts = [prompts.strip()]

        start_time = time.time()
        result_images = pipeline(
            prompt=prompts[0],
            num_images_per_prompt=args.batch_size,
            num_inference_steps=args.infer_steps,
            seed_generator=seed_generator,
        )[0]
        pipeline_time = time.time() - start_time
        logger.info("HunyuanDiT pipeline_time: %.3f", pipeline_time)
        torch.npu.empty_cache()

        for i, img in enumerate(result_images):
            save_path = os.path.join(time_path, f"0_{i}.png")
            img.save(save_path)
            torch.npu.empty_cache()


if __name__ == "__main__":
    inference_args = parse_arguments()
    if not os.path.exists(inference_args.path):
        raise ValueError(f"The model path not exists: {inference_args.path}")

    torch.npu.set_device(inference_args.device_id)
    infer(inference_args)