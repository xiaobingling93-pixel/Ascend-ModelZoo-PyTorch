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
import argparse
import time
import logging

import torch
import torch_npu
import torch.distributed as dist

from juggernaut_xi_lightning import StableDiffusionXLPipeline, UNet2DConditionModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='./Juggernaut_XI_Lightning',
        help="The path of all model weights, suach as vae, unet, text_encoder, tokenizer, scheduler",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts.txt",
        help="The prompts file to guide audio generation.",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="NPU device id",
    )
    parser.add_argument(
        "--dtype",
        type=torch.dtype,
        default=torch.float16
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="The dir to save image",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Random seed, default 66",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6,
        help="Random seed, default 66",
    )
    parser.add_argument(
        "--use_parallel",
        action="store_true",
        help="Turn on dual parallel",
    )
    parser.add_argument(
        "--cache_method",
        type=str,
        default="",
        help="support agb_cache method only",
    )
    return parser.parse_args()


def init_process():
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend='hccl',
        init_method='env://', 
        world_size=world_size,
        rank=rank,
    )


def test_performance():
    args = parse_arguments()
    if args.use_parallel:
        init_process()
    else:
        torch.npu.set_device(args.device_id)
    torch.manual_seed(args.seed)

    unet = UNet2DConditionModel.from_pretrained(args.path, subfolder='unet', cache_method=args.cache_method)
    pipe = StableDiffusionXLPipeline.from_pretrained(args.path, unet=unet)
    pipe.to(args.dtype).to("npu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_time = 0
    prompts_num = 0
    skip = 3
    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        for i, prompt in enumerate(f):
            with torch.no_grad():
                begin_time = time.time()
                image = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_images_per_prompt=args.num_images_per_prompt,
                    num_inference_steps=args.steps,
                    use_parallel=args.use_parallel,
                )[0]
                if i > skip - 1: # skip the first 3 infer.
                    end_time = time.time()
                    all_time += (end_time - begin_time)
            prompts_num += 1
            image[0].save(os.path.join(args.output_dir, f"image_{i}.png"))
    if prompts_num >= 3:
        logger.info(f"Time interval is {all_time / (prompts_num - skip)}") # skip the first 3 infer.
    else:
        raise ValueError("Infer average time skip first two prompts, ensure that prompts.txt \
                         contains more than three prompts")


if __name__ == "__main__":
    test_performance()