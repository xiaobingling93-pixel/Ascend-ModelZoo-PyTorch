import os
import argparse
import time
import functools

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video

from typing import List, Optional, Tuple, Union, Literal

from cogvideox_5b import CogVideoXPipeline, CogVideoXTransformer3DModel, get_rank, get_world_size, all_gather


def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: torch.LongTensor = None,
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        temporal_size = hidden_states.shape[1]
        n, c, t, h, w = hidden_states.shape
        hidden_states = torch.cat([hidden_states, torch.zeros(n, c, t, 4, w, device=hidden_states.device, dtype=hidden_states.dtype)], dim=-2)
        hidden_states = torch.chunk(hidden_states, get_world_size(), dim=-2)[get_rank()]
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb

            def get_rotary_emb_chunk(freqs):
                dim_thw = freqs.shape[-1]
                freqs = freqs.reshape(temporal_size, -1, dim_thw)
                freqs = freqs.reshape(temporal_size, -1, 45, dim_thw)
                freqs = torch.cat([freqs, torch.zeros(temporal_size, 2, 45, dim_thw, device=freqs.device, dtype=freqs.dtype)], dim=1)
                freqs = freqs.reshape(temporal_size, -1, dim_thw)
                freqs = torch.chunk(freqs, get_world_size(), dim=-2)[get_rank()]
                freqs = freqs.reshape(-1, dim_thw)
                return freqs

            freqs_cos = get_rotary_emb_chunk(freqs_cos)
            freqs_sin = get_rotary_emb_chunk(freqs_sin)
            image_rotary_emb = (freqs_cos, freqs_sin)
        
        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            timestep=timestep,
            timestep_cond=timestep_cond,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = all_gather(sample, dim=-2)
        sample = sample[:, :, :, :-4, :]
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
    
    original_patch_embed_forward = transformer.patch_embed.forward
    
    @functools.wraps(transformer.patch_embed.__class__.forward)
    def new_patch_embed(
        self, text_embeds: torch.Tensor, image_embeds: torch.Tensor
    ):
        text_embeds = all_gather(text_embeds.contiguous(), dim=-2)
        image_embeds = all_gather(image_embeds.contiguous(), dim=-2)
        batch, num_frames, channels, height, width = image_embeds.shape
        text_len = text_embeds.shape[-2]
        output = original_patch_embed_forward(text_embeds, image_embeds)
        text_embeds = output[:, :text_len, :]
        image_embeds = output[:, text_len:, :].reshape(batch, num_frames, -1, output.shape[-1])

        text_embeds = torch.chunk(text_embeds, get_world_size(), dim=-2)[get_rank()]
        image_embeds = torch.chunk(image_embeds, get_world_size(), dim=-2)[get_rank()]
        image_embeds = image_embeds.reshape(batch, -1, image_embeds.shape[-1])
        return torch.cat([text_embeds, image_embeds], dim=1)

    new_patch_embed = new_patch_embed.__get__(transformer.patch_embed)
    transformer.patch_embed.forward = new_patch_embed


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 8
):
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype).to(f"npu:{get_rank()}")
    transformer = CogVideoXTransformer3DModel.from_pretrained(os.path.join(model_path, 'transformer'), torch_dtype=dtype).to(f"npu:{get_rank()}")
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.transformer = transformer
    pipe.vae = pipe.vae.half()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.switch_to_qkvLinear()
    if get_world_size() > 1:
        parallelize_transformer(pipe)

    # warm up
    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=1,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        output_type="pil"
    ).frames[0]

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

    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
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
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
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

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.npu.config.allow_internal_format = False
    generate_video(
        prompt=args.prompt,
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
    )
