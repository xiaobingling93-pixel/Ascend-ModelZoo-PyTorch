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
import torch
import numpy as np

import torch_npu
from torch_npu.contrib import transfer_to_npu

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="StableDiffusion and ChatGLM infer args")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--output_path", type=str, default="./output", help="Output directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--device_map", type=str, required=True, choices=["cuda", "npu", "cpu", "auto"],
                        help="The device to conduct inference")
    parser.add_argument("--seed", type=int, default=66, help="Random seed")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    return parser.parse_args()


def infer(args):
    ckpt_dir = args.ckpt_dir
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False)

    pipe = pipe.to(args.device_map)
    pipe.enable_model_cpu_offload()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Generate image
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator=torch.Generator(pipe.device).manual_seed(args.seed)
    ).images[0]

    # Save the generated image
    output_image_path = os.path.join(args.output_path, 'sample_test.jpg')
    image.save(output_image_path)
    print(f"Image saved to {output_image_path}")


if __name__ == '__main__':
    args = parse_args()
    infer(args)