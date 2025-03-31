import os
import time
from pathlib import Path
from loguru import logger

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu 
import cv2
import numpy as np
from tqdm import tqdm
from mindiesd.runtime.cache_manager import CacheManager, DitCacheConfig
from msmodelslim.pytorch.multimodal import DitCacheSearcherConfig, DitCacheSearcher
torch_npu.npu.set_compile_mode(jit_compile=False) 
torch.npu.config.allow_internal_format = False


def run_model_and_save_samples(config: DitCacheSearcherConfig, hunyuan_video_sampler):
    # 通过config的成员对pipeline.transformer成员变量赋值
    dit_cache_config = DitCacheConfig(
        step_start=config.cache_step_start,
        step_interval=config.cache_step_interval,
        block_start=config.cache_dit_block_start,
        num_blocks=config.cache_num_dit_blocks
    )
    cache = CacheManager(dit_cache_config)
    if args.search_single_cache:
        pipeline.transformer.cache_single = cache
    elif args.search_double_cache:
        pipeline.transformer.cache_dual = cache

    prompts = [
        "realistic style, a lone cowboy rides his horse across an open plain at beautiful sunset, soft light, warm colors",
        "extreme close-up with a shallow depth of field of a puddle in a street. reflecting a busy futuristic Tokyo city with bright neon signs, night, lens flare",
        "Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. Shallow focus and light smoke. vivid colours",
        "Timelapse of the northern lights dancing across the Arctic sky, stars twinkling, snow-covered landscape",
        "A panning shot of a serene mountain landscape, slowly revealing snow-capped peaks, granite rocks and a crystal-clear lake reflecting the sky",
        "moody shot of a central European alley film noir cinematic black and white high contrast high detail"
    ]
    for idx, prompt_ in enumerate(prompts):
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
    
        if config.cache_num_dit_blocks == 0:
            video_path = os.path.join(args.save_path, f'sample_{idx:04d}_no_cache.mp4')
        else:
            video_path = os.path.join(args.save_path,
                    f'sample_{idx:04d}_{config.cache_dit_block_start}_{config.cache_step_interval}_{config.cache_num_dit_blocks}_{config.cache_step_start}.mp4')
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for j, sample in enumerate(samples):
                sample = samples[j].unsqueeze(0)
                save_videos_grid(sample, video_path, fps=24)
                logger.info(f'Sample save to: {video_path}')


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


def search_single_block():
    block_num = len(pipeline.transformer.single_blocks)
    print(f"single block num:{block_num}")

    config = DitCacheSearcherConfig(
        dit_block_num=block_num,
        prompts_num=1,
        num_sampling_steps=args.infer_steps,
        cache_ratio=args.cache_ratio,
        search_cache_path=args.save_path,
        cache_step_start=0,
        cache_dit_block_start=0,
        cache_num_dit_blocks=0
    )
    search_handler = DitCacheSearcher(config, hunyuan_video_sampler, run_model_and_save_samples)
    cache_final_list = search_handler.search()
    print(f'****************single block: cache_final_list in \
          [cache_dit_block_start, cache_step_interval, cache_num_dit_blocks, cache_step_start] order: {cache_final_list}')


def search_double_block():
    block_num = len(pipeline.transformer.double_blocks)
    print(f"double block num:{block_num}")

    config = DitCacheSearcherConfig(
        dit_block_num=block_num,
        prompts_num=1,
        num_sampling_steps=args.infer_steps,
        cache_ratio=args.cache_ratio,
        search_cache_path=args.save_path,
        cache_step_start=0,
        cache_dit_block_start=0,
        cache_num_dit_blocks=0
    )
    search_handler = DitCacheSearcher(config, hunyuan_video_sampler, run_model_and_save_samples)
    cache_final_list = search_handler.search()
    print(f'****************double block: cache_final_list in \
          [cache_dit_block_start, cache_step_interval, \
          cache_num_dit_blocks, cache_step_start] order: {cache_final_list}')


def check_args(args):
    if args.search_single_cache and args.search_double_cache:
        print("Cache policies cannot be searched at the same time.")
        return False
    if (not args.search_single_cache) and (not args.search_double_cache):
        print("Must specify search_single_cache or search_double_cache parameter.")
        return False
    if args.search_single_cache:
        if not args.use_cache:
            print(f"If you want to use the 'search_single_cache' parameter, \
            please specify the 'use_cache' parameter at the same time.")
            return False
    if args.search_double_cache:
        if not args.use_cache_double:
            print(f"If you want to use the 'search_double_cache' parameter, \
            please specify the 'use_cache_double' parameter at the same time.")
            return False
    return True


if __name__ == "__main__":
    hunyuan_video_sampler = prepare_pipeline()
    args = hunyuan_video_sampler.args
    pipeline = hunyuan_video_sampler.pipeline
    args_status = check_args(args)

    if args_status:
        if args.search_single_cache:
            search_single_block()
        if args.search_double_cache:
            search_double_block()
        print("Search Done!!!!")