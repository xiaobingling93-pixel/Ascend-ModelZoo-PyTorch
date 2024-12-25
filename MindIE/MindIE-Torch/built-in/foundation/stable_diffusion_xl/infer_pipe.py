import os
import argparse
import time
from typing import Union, List
import torch
import torch_npu
import numpy as np
import cv2

from stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from stable_diffusion_xl.unet.unet_model import UNet2DConditionModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='/stable-diffusion-xl-base-1.0',
        help="The path of all model weights, suach as vae, unet, text_encoder, tokenizer, scheduler",
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
        "--prompts",
        type=List[str],
        default=["A dog, site on beach."]
    )
    parser.add_argument(
        "--num_image_per_prompt",
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
    return parser.parse_args()


def init_env(device_id: int):
    torch.npu.set_device(device_id)


def init_pipe(model_path: str, dtype=torch.float16):
    unet = UNet2DConditionModel.from_pretrained(os.path.join(model_path, 'unet'), cache_method="agb_cahce")
    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, unet=unet)
    pipe.to(dtype).to("npu")
    return pipe


def infer_prompts(pipe, prompts, height=1024, width=1024, num_image=1):
    images = pipe(
        prompt=prompts,
        height=height,
        width=width,
        num_images_per_prompt=num_image
    ).images
    for i, img in enumerate(images):
        img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{i}.png", img_bgr)


if __name__ == "__main__":
    args = parse_arguments()
    init_env(args.device_id)
    pipe = init_pipe(args.path, args.dtype)
    prompts = args.prompts
    infer_prompts(pipe, prompts, args.height, args.width, args.num_image_per_prompt)
