import os
import time
import math
import logging
from typing import List
from pathlib import Path
import cv2
import numpy as np
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from attentioncache_search_tool import CacheSearcher
from mindiesd import CacheConfig, CacheAgent
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Cache Searcher")


def prepare_pipeline():
    args_init = parse_args()
    models_root_path = Path(args_init.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args_init.save_path if args_init.save_path_suffix == "" else f'{args_init.save_path}_{args_init.save_path_suffix}'
    if not os.path.exists(args_init.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args_init)
    return hunyuan_video_sampler


def generate_videos(prompts, hunyuan_video_sampler):
    # prompts, num_sampling_steps, config, self.pipeline
    # 通过config的成员对pipeline.transformer成员变量赋值

    output_list = []
    for prompt_ in prompts:
        # Start sampling
        outputs = hunyuan_video_sampler.predict(
                prompt=prompt_, 
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale
            )
        samples = outputs['samples']

        for j, _ in enumerate(samples):
            output_list.append(samples[j])
    return output_list

if __name__ == "__main__":
    hunyuan_video_sampler = prepare_pipeline()
    args = hunyuan_video_sampler.args
    pipeline = hunyuan_video_sampler.pipeline
    transformer = hunyuan_video_sampler.pipeline.transformer

    prompts = [
        "realistic style, a lone cowboy rides his horse across an open plain at beautiful sunset, soft light, warm colors",
        "extreme close-up with a shallow depth of field of a puddle in a street. reflecting a busy futuristic Tokyo city with bright neon signs, night, lens flare"
    ]

    config = CacheConfig(
        method="attention_cache",
        blocks_count=len(transformer.double_blocks + transformer.single_blocks),
        steps_count=args.infer_steps
    )
    cache = CacheAgent(config)
    for block in transformer.double_blocks + transformer.single_blocks:
        block.cache = cache
    
    search = CacheSearcher(cache)
    search.search_apply(
        args.infer_steps,
        2.0,
        generate_videos,
        prompts=prompts,
        hunyuan_video_sampler=hunyuan_video_sampler
    )