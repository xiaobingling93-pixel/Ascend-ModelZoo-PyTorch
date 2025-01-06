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

import torch

from cogview3plus import CogView3PlusPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an image using the CogView3-Plus-3B model.")

    # Define arguments for prompt, model path, etc.
    parser.add_argument(
        "--prompt", 
        type=list, 
        default=[
            "A vibrant cherry red sports car sits proudly under the gleaming sun, \
            its polished exterior smooth and flawless, casting a mirror-like reflection. \
            The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, \
            and a set of black, high-gloss racing rims that contrast starkly with the red. \
            A subtle hint of chrome embellishes the grille and exhaust, \
            while the tinted windows suggest a luxurious and private interior. \
            he scene conveys a sense of speed and elegance, \
            the car appearing as if it's about to burst into a sprint along a coastal road, \
            with the ocean's azure waves crashing in the background."
        ], 
        help="The text description for generating the image."
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
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps for inference.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--output_path", type=str, default="cogview3.png", help="Path to save the generated image.")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16 or fp16")
    parser.add_argument("--device_id", type=int, default=7, help="NPU device id")

    return parser.parse_args()


def infer(args):
    torch.npu.set_device(args.device_id)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Load the pre-trained model with the specified precision
    pipe = CogView3PlusPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to("npu")

    use_time = 0
    loops = 5
    for i in range(loops):
        start_time = time.time()
        # Generate the image based on the prompt
        image = pipe(
            prompt=args.prompt[0],
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            num_inference_steps=args.num_inference_steps,
            image_size=(args.height, args.width),
        ).images[0]
        
        if i >= 2:
            use_time += time.time() - start_time
            logger.info("current_time is %.3f )", time.time() - start_time)

        torch.npu.empty_cache()
    
    logger.info("use_time is %.3f)", use_time / 3)

    # Save the generated image to the local file system
    image.save(args.output_path)

    print(f"Image saved to {args.output_path}")


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)

