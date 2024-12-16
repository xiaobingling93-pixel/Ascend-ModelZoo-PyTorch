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
import argparse
import time
import logging
import colossalai
import torch
import torch.distributed as dist
from torchvision.io import write_video

from opensora import set_parallel_manager
from opensora import compile_pipe
from opensora import OpenSoraPipeline12

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='/open-sora',
        help="The path of all model weights, suach as vae, transformer, text_encoder, tokenizer, scheduler",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="NPU device id",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='npu',
        help="NPU",
    )
    parser.add_argument(
        "--type",
        type=str,
        default='bf16',
        help="bf16 or fp16",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="num_frames: 32 or 128",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="(720, 1280)",
        help="image_size: (720, 1280) or (512, 512)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="fps: 8",
    )
    parser.add_argument(
        "--enable_sequence_parallelism",
        type=bool,
        default=False,
        help="enable_sequence_parallelism",
    )
    parser.add_argument(
        "--set_patch_parallel",
        type=bool,
        default=False,
        help="set_patch_parallel",
    )
    parser.add_argument(
        "--prompts",
        type=list,
        default=[
            'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. \
             She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. \
             She wears sunglasses and red lipstick. She walks confidently and casually. \
             The street is damp and reflective, creating a mirror effect of the colorful lights. \
             Many pedestrians walk about.'],
        help="prompts",
    )
    parser.add_argument(
        "--test_acc",
        action="store_true",
        help="Run or not.",
    )
    return parser.parse_args()


def infer(args):
    test_acc = args.test_acc
    use_time = 0
    torch.npu.set_device(args.device_id)
    dtype = torch.bfloat16
    if args.type == 'bf16':
        dtype = torch.bfloat16
    elif args.type == 'fp16':
        dtype = torch.float16
    else:
        logger.error("Not supported.")

    # === Initialize Distributed ===
    if args.enable_sequence_parallelism or args.set_patch_parallel:
        colossalai.launch_from_torch({})
        sp_size = dist.get_world_size()
        set_parallel_manager(sp_size, sp_axis=0)

    args.image_size = eval(args.image_size)

    if not test_acc:
        prompts = args.prompts
    else:
        lines_list = []
        with open('./prompts/t2v_sora.txt', 'r') as file:
            for line in file:
                line = line.strip()
                lines_list.append(line)
        prompts = lines_list

    if not test_acc:
        loops = 5
    else:
        loops = len(prompts)

    pipe = OpenSoraPipeline12.from_pretrained(model_path=args.path,
                                              num_frames=args.num_frames, image_size=args.image_size, fps=args.fps,
                                              enable_sequence_parallelism=args.enable_sequence_parallelism,
                                              dtype=dtype, openmind_name="opensora_v1_2")
    pipe = compile_pipe(pipe)

    for i in range(loops):

        start_time = time.time()
        if test_acc:
            video = pipe(prompts=[prompts[i]], output_type="thwc")

        else:
            video = pipe(prompts=prompts)

        torch.npu.empty_cache()

        if test_acc:
            if i < 10:
                save_file_name = "sample_0{}.mp4".format(i)
            else:
                save_file_name = "sample_{}.mp4".format(i)
            save_path = os.path.join(os.getcwd(), save_file_name)

            write_video(save_path, video, fps=8, video_codec="h264")
            torch.npu.empty_cache()
        else:
            if i >= 2:
                use_time += time.time() - start_time
                logger.info("current_time is %.3f )", time.time() - start_time)
        torch.npu.empty_cache()

    if not test_acc:
        logger.info("use_time is %.3f)", use_time / 3)


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)

