import os
import time
from pathlib import Path
from loguru import logger
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from mindiesd import CacheConfig, CacheAgent
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


def main():
    args = parse_args()
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    transformer = hunyuan_video_sampler.pipeline.transformer
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    if args.prompt.endswith('txt'):
        with open(args.prompt, 'r') as file:
            text_prompt = file.readlines()
            prompts = [line.strip() for line in text_prompt]
    else:
        prompts = [args.prompt]

    if args.use_cache:
        # single
        config_single = CacheConfig(
            method="dit_block_cache",
            blocks_count=len(transformer.single_blocks),
            steps_count=args.infer_steps,
            step_start=args.cache_start_steps,
            step_interval=args.cache_interval,
            step_end=args.infer_steps - 1,
            block_start=args.single_block_start,
            block_end=args.single_block_end
        )
        cache_single = CacheAgent(config_single)
        hunyuan_video_sampler.pipeline.transformer.cache_single = cache_single
    if args.use_cache_double:
        # double
        config_double = CacheConfig(
            method="dit_block_cache",
            blocks_count=len(transformer.double_blocks),
            steps_count=args.infer_steps,
            step_start=args.cache_start_steps,
            step_interval=args.cache_interval,
            step_end=args.infer_steps - 1,
            block_start=args.double_block_start,
            block_end=args.double_block_end
        )
        cache_dual = CacheAgent(config_double)
        hunyuan_video_sampler.pipeline.transformer.cache_dual = cache_dual

    if args.use_attentioncache:
        config_double = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.double_blocks),
            steps_count=args.infer_steps,
            step_start=args.start_step,
            step_interval=args.attentioncache_interval,
            step_end=args.end_step
        )
        config_single = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.single_blocks),
            steps_count=args.infer_steps,
            step_start=args.start_step,
            step_interval=args.attentioncache_interval,
            step_end=args.end_step
        )
    else:
        config_double = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.double_blocks),
            steps_count=args.infer_steps
        )
        config_single = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.single_blocks),
            steps_count=args.infer_steps
        )
    cache_double = CacheAgent(config_double)
    cache_single = CacheAgent(config_single)
    for block in transformer.double_blocks:
        block.cache = cache_double
    for block in transformer.single_blocks:
        block.cache = cache_single

    # warmup
    outputs = hunyuan_video_sampler.predict(
        prompt=prompts[0], 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=2,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    # Start sampling
    for idx, prompt_ in enumerate(prompts):
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
        
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                video_path = f"{save_path}/sample_{idx}.mp4"
                save_videos_grid(sample, video_path, fps=24)
                logger.info(f'Sample save to: {video_path}')

if __name__ == "__main__":
    main()