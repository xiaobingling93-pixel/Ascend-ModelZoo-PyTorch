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
import sys
import time
import argparse
import logging

import torch
import torch_npu
from torchvision.utils import save_image
import imageio
import colossalai

sys.path.append(os.path.split(sys.path[0])[0])

from opensoraplan import OpenSoraPlanPipeline
from opensoraplan import compile_pipe, get_scheduler, set_parallel_manager
from opensoraplan import CacheConfig, OpenSoraPlanDiTCacheManager

MASTER_PORT = '42043'


def main(args):
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    device = "npu" if torch.npu.is_available() else "cpu"

    sp_size = args.sequence_parallel_size
    if sp_size == 1:
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = MASTER_PORT
    colossalai.launch_from_torch({}, seed=args.seed)
    set_parallel_manager(sp_size=args.sequence_parallel_size, sp_axis=0)

    if args.force_images:
        ext = 'jpg'
    else:
        ext = 'mp4'
    scheduler = get_scheduler(args.sample_method)
    # load the pipeline model weights and config
    videogen_pipeline = OpenSoraPlanPipeline.from_pretrained(model_path=args.model_path,
                                                             image_size=args.image_size,
                                                             scheduler=scheduler,
                                                             dtype=torch.float16,
                                                             vae_stride=args.vae_stride)
    # prepare the cache_manager
    cache_nums = [int(i) for i in args.cache_config.split(',')]
    if len(cache_nums) != 4:
        raise ValueError("cache_config num length must equals 4.")
    cache_manager = OpenSoraPlanDiTCacheManager(
        CacheConfig(cache_nums[0], cache_nums[1], cache_nums[2], cache_nums[3], args.use_cache))
    # compile pipeline and set the cache_manager and cfg_last_step
    videogen_pipeline = compile_pipe(videogen_pipeline, cache_manager, args.cfg_last_step)

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    # read the prompt contents
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]
        args.text_prompt = args.text_prompt

    time_list = []
    # pipeline inference
    for idx, prompt in enumerate(args.text_prompt):
        torch_npu.npu.synchronize()
        start_time = time.time()
        torch.manual_seed(args.seed)
        torch.npu.manual_seed(args.seed)
        torch.npu.manual_seed_all(args.seed)
        logging.info('Processing the (%s) prompt', prompt)
        videos = videogen_pipeline(prompt,
                                   num_inference_steps=args.num_sampling_steps,
                                   guidance_scale=args.guidance_scale,
                                   enable_temporal_attentions=not args.force_images,
                                   num_images_per_prompt=1,
                                   ).video
        if videogen_pipeline.transformer.cache_manager.cal_block_num != 0:
            ratio = (
                    videogen_pipeline.transformer.cache_manager.all_block_num
                    / videogen_pipeline.transformer.cache_manager.cal_block_num
            )
        else:
            raise ZeroDivisionError("transformer cal_block_num can not be zero.")
        logging.info("cal_block_ratio: %.2f, %d, %d",
                     ratio, videogen_pipeline.transformer.cache_manager.cal_block_num,
                     videogen_pipeline.transformer.cache_manager.all_block_num)
        torch_npu.npu.synchronize()
        time_list.append(time.time() - start_time)
        try:
            if args.force_images:
                videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                save_image(
                    videos / 255.0,
                    os.path.join(
                        args.save_img_path,
                        prompt.replace(' ', '_')[:100] +
                        f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}',
                    ),
                    nrow=1, normalize=True, value_range=(0, 1)
                )  # t c h w
            else:
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        f'sample_{idx}_{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
                    ), videos[0],
                    fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
            logging.info('Saving sample_%d for %s %d steps success!!!', \
                         idx, args.sample_method, args.num_sampling_steps)
        except IOError as e:
            logging.error('Error when saving sample_%d for %s %d steps for %s!!!', \
                          idx, args.sample_method, args.num_sampling_steps, prompt)
            sys.exit('An error occured and the program will exit.')

    logging.info("time_list: %s", time_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/data1/models/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--vae_stride", type=int, default=8)
    parser.add_argument("--cache_config", type=str, default="5,27,5,2")
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument("--cfg_last_step", type=int, default=10000)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--force_images', action='store_true')
    parser.add_argument('--sequence_parallel_size', type=int, default=1)
    args_input = parser.parse_args()

    if not os.path.exists(args_input.model_path):
        logging.warning('WARNING:wrong model_path given !!!')
        sys.exit('An error occured and the program will exit.')

    main(args_input)
