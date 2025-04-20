import time
import json
import os
import argparse

import torch
import torch_npu
import soundfile as sf
from safetensors.torch import load_file

from stableaudio import (
    StableAudioPipeline,
    AutoencoderOobleck,
)
from mindiesd import CacheConfig, CacheAgent


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts/prompts.txt",
        help="The prompts file to guide audio generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="The prompt or prompts to guide what to not include in audio generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./stable-audio-open-1.0",
        help="The path of stable-audio-open-1.0.",
    )
    parser.add_argument(
        "--audio_end_in_s",
        nargs='+',
        default=[10],
        help="Audio end index in seconds.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device id.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result audio files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed, default 1.",
    )
    parser.add_argument(
        "--use_ditcache",
        action="store_true",
        help="turn on ditcache or not.",
    )
    parser = add_attentioncache_args(parser)
    return parser.parse_args()


def add_attentioncache_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Attention Cache args")
    group.add_argument("--use_attentioncache", action='store_true')
    group.add_argument("--attentioncache_ratio", type=float, default=1.4)
    group.add_argument("--attentioncache_interval", type=int, default=5)
    group.add_argument("--start_step", type=int, default=60)
    group.add_argument("--end_step", type=int, default=97)

    return parser


def main():
    args = parse_arguments()
    if (args.use_ditcache and args.use_attentioncache):
        raise ValueError(f"Only support one cache at a time, but got use_ditcache is {args.use_ditcache}, and use_attentioncache is {args.use_attentioncache}")
        
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    npu_stream = torch_npu.npu.Stream()
    torch_npu.npu.set_device(args.device)
    if args.seed != -1:
        torch.manual_seed(args.seed)
    latents = torch.randn(1, 64, 1024, dtype=torch.float16, device="cpu").to("npu")

    with open(os.path.join(args.model, "vae", "config.json")) as f:
        vae_config = json.load(f)
    vae = AutoencoderOobleck.from_config(vae_config)
    vae.load_state_dict(load_file(os.path.join(args.model, "vae", "diffusion_pytorch_model.safetensors")))
    
    pipe = StableAudioPipeline.from_pretrained(args.model, vae=vae)
    pipe.to(torch.float16).to("npu")

    transformer = pipe.transformer
    if args.use_attentioncache:
        config = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.transformer_blocks),
            steps_count=args.num_inference_steps,
            step_start=args.start_step,
            step_interval=args.attentioncache_interval,
            step_end=args.end_step
        )
    else:
        config = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.transformer_blocks),
            steps_count=args.num_inference_steps
        )
    cache = CacheAgent(config)
    for block in transformer.transformer_blocks:
        block.cache = cache

    total_time = 0
    prompts_num = 0
    average_time = 0
    skip = 2
    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        for i, prompt in enumerate(f):
            with torch.no_grad():
                audio_end_in_s = float(args.audio_end_in_s[i]) if (len(args.audio_end_in_s) > i) else 10.0
                npu_stream.synchronize()
                begin = time.time()
                audio = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    latents=latents,
                    audio_end_in_s=audio_end_in_s,
                    use_cache=args.use_ditcache,
                ).audios
                npu_stream.synchronize()
                end = time.time()
                if i > skip - 1:
                    total_time += end - begin
            prompts_num = i + 1
            output = audio[0].T.float().cpu().numpy()
            file_path = os.path.join(args.save_dir, f"audio_by_prompt{prompts_num}.wav")
            sf.write(file_path, output, pipe.vae.sampling_rate)
    if prompts_num > skip:
        average_time = total_time / (prompts_num - skip)
    else:
        raise ValueError("Infer average time skip first two prompts, ensure that prompts.txt \
                         contains more than three prompts")
    print(f"Infer average time: {average_time:.3f}s\n")

if __name__ == "__main__":
    main()