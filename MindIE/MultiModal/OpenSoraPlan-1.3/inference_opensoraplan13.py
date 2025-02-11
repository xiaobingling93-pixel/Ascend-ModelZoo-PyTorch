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

import os
import argparse
import time
import logging

import torch 
import torch_npu
import imageio

from transformers import AutoTokenizer, MT5EncoderModel
from open_sora_planv1_3.pipeline.open_soar_plan_pipeline import OpenSoraPlanPipeline13
from open_sora_planv1_3.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from open_sora_planv1_3.models.t2vdit import OpenSoraT2Vv1_3
from open_sora_planv1_3.models.wfvae import WFVAEModelWrapper, ae_stride_config
from open_sora_planv1_3.utils import set_random_seed
from open_sora_planv1_3.models.parallel_mgr import init_parallel_env, get_sequence_parallel_rank
from open_sora_planv1_3.layers.cache_mgr import CacheManager, DitCacheConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test Pipeline Argument Parser')

    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--version', type=str, default='v1_3', help='Version of the model')
    parser.add_argument('--dtype', type=str, default='fp16', help='Data type used in inference')
    parser.add_argument('--num_frames', type=int, default=93, help='Number of frames')
    parser.add_argument('--height', type=int, default=720, help='Height of the frames')
    parser.add_argument('--width', type=int, default=1280, help='Width of the frames')
    parser.add_argument('--text_encoder_name_1', type=str, required=True, help='Path to the text encoder model')
    parser.add_argument('--text_prompt', type=str, required=True, help='Text prompt for the model')
    parser.add_argument('--ae', type=str, default='WFVAEModel_D8_4x8x8', help='Autoencoder model type')
    parser.add_argument('--ae_path', type=str, required=True, help='Path to the autoencoder model')
    parser.add_argument('--save_img_path', type=str, default='./test', help='Path to save images')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for the model')
    parser.add_argument('--num_sampling_steps', type=int, default=10, help='Number of sampling steps')
    parser.add_argument('--max_sequence_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_samples_per_prompt', type=int, default=1, help='Number of samples per prompt')
    parser.add_argument('--rescale_betas_zero_snr', action='store_true', help='Rescale betas zero SNR')
    parser.add_argument('--prediction_type', type=str, default='v_prediction', help='Type of prediction')
    parser.add_argument('--save_memory', action='store_true', help='Save memory during processing')
    parser.add_argument('--enable_tiling', action='store_true', help='Enable tiling for processing')
    parser.add_argument('--sp', action='store_true')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--cache_sampling_step_start', type=int, default=20, help='Sampling step begins to use cache')
    parser.add_argument('--cache_sampling_step_interval', type=int, default=2, help='Sampling step interval of cache')
    parser.add_argument('--cache_dit_block_start', type=int, default=2, help='DiT block id begins to be cached')
    parser.add_argument('--cache_num_dit_blocks', type=int, default=20, help='DiT blocks cached in each step')
    args = parser.parse_args()
    return args


def infer(args):
    dtype = torch.bfloat16
    if args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        dtype = torch.float16
    else:
        logger.error("Not supported.")
    # === Initialize Distributed === 
    init_parallel_env(args.sp)

    set_random_seed(args.seed + get_sequence_parallel_rank())

    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """
    positive_prompt = """
    high quality, high aesthetic, {}
    """
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]

    vae = WFVAEModelWrapper.from_pretrained(args.ae_path, dtype=torch.float16).to("npu").eval()
    vae.vae_scale_factor = ae_stride_config[args.ae]
    transformer = OpenSoraT2Vv1_3.from_pretrained(args.model_path).to(dtype).to("npu").eval()

    kwargs = dict(
        prediction_type=args.prediction_type, 
        rescale_betas_zero_snr=args.rescale_betas_zero_snr, 
        timestep_spacing="trailing" if args.rescale_betas_zero_snr else 'leading', 
    )
    scheduler = EulerAncestralDiscreteScheduler(**kwargs)
    text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name_1, 
                                                   torch_dtype=dtype).eval().to(dtype).to("npu")
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name_1)

    if args.save_memory:
        vae.vae.enable_tiling()
        vae.vae.t_chunk_enc = 8
        vae.vae.t_chunk_dec = 2 
            
    pipeline = OpenSoraPlanPipeline13(vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler)

    if args.use_cache:
        config = DitCacheConfig(step_start=20, step_interval=2, block_start=2, num_blocks=20)
        cache = CacheManager(config)
        pipeline.transformer.cache = cache
    
    with torch.no_grad():
        for i, input_prompt in enumerate(args.text_prompt):
            input_prompt = positive_prompt.format(input_prompt)
            start_time = time.time()
            videos = pipeline(
                        input_prompt, 
                        negative_prompt=negative_prompt, 
                        num_frames=args.num_frames,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_sampling_steps,
                        guidance_scale=args.guidance_scale,
                        num_samples_per_prompt=args.num_samples_per_prompt,
                        max_sequence_length=args.max_sequence_length,
                    )[0]
            torch.npu.synchronize()
            use_time = time.time() - start_time
            logger.info("use_time: %.3f", use_time)
            imageio.mimwrite(
                os.path.join(
                    args.save_img_path,
                    f's{args.num_sampling_steps}_prompt{i}.mp4'
                ), 
                videos[0],
                fps=args.fps, 
                quality=6
                )  # highest quality is 10, lowest is 0

if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)