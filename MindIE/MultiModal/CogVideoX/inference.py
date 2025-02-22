import os
import json
import argparse
import time
import random

from typing import Literal

import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video

from cogvideox_5b import CogVideoXPipeline, CogVideoXTransformer3DModel, get_rank, get_world_size, all_gather, set_parallel
from mindiesd.pipeline.sampling_optm import AdaStep


def generate_video(
    prompt_file: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    output_path: str = "./output",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 8,
    enable_skip: bool = True
):
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True).to(f"npu:{get_rank()}")
    transformer = CogVideoXTransformer3DModel.from_pretrained(os.path.join(model_path, 'transformer'), torch_dtype=dtype, local_files_only=True).to(f"npu:{get_rank()}")
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.transformer = transformer
    pipe.vae = pipe.vae.half()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.switch_to_qkvLinear()
    set_parallel(pipe)

    # sampling optm
    if enable_skip and transformer.config.use_rotary_positional_embeddings:
        skip_strategy = AdaStep(skip_thr=0.006, max_skip_steps=1, decay_ratio=0.99, device="npu")
        pipe.skip_strategy = skip_strategy
    elif enable_skip and not transformer.config.use_rotary_positional_embeddings:
        skip_strategy = AdaStep(skip_thr=0.009, max_skip_steps=1, decay_ratio=0.99, device="npu")
        pipe.skip_strategy = skip_strategy

    # warm up
    video_generate = pipe(
        height=height,
        width=width,
        prompt="A dog",
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=1,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        output_type="pil"
    ).frames[0]

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"The file {prompt_file} does not exist.")
    
    result = {}
    
    with open(prompt_file, 'r', encoding='utf-8') as file:
        prompts = file.readlines()

    os.makedirs(output_path, exist_ok=True)
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()  # 去掉可能的空格和换行符
        torch_npu.npu.synchronize()
        start = time.time()
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            output_type="pil"
        ).frames[0]
        torch_npu.npu.synchronize()
        end = time.time()
        print(f"Time taken for inference: {end - start} seconds")
        if enable_skip and not transformer.config.use_rotary_positional_embeddings:
            skip_strategy = AdaStep(skip_thr=0.009, max_skip_steps=1, decay_ratio=0.99, device="npu")
            pipe.skip_strategy = skip_strategy
        
        video_path = f'{output_path}/generated_video_{i}_{prompt[:10]}.mp4'
        export_to_video(video_generate, video_path, fps=fps)
        result[os.path.abspath(video_path)] = prompt

    with open(f'{output_path}/result.json', 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)
    
    print(f"Result saved to result.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt_file", type=str, default="./prompts.txt", help="The prompt file")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="/data/CogVideoX-5b", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=48, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=720, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=480, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=8, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument('--enable_skip', action='store_true', help='enable_skip')

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.npu.config.allow_internal_format = False
    generate_video(
        prompt_file=args.prompt_file,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
        enable_skip=args.enable_skip
    )
