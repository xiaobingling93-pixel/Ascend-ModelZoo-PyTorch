# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import numpy as np
from modelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig
from quant_utils import modify_model
import argparse
from argparse import Namespace
import math
import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from diffusers import StableDiffusionXLPipeline
from compile_model import *


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save pt models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./stable-diffusion-xl-base-1.0",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("-steps", "--steps", type=int, default=50, help="steps.")
    parser.add_argument("-guid", "--guidance_scale", type=float, default=5.0, help="guidance_scale")
    parser.add_argument("--use_cache", action="store_true", help="Use cache during inference.")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="Export the unet of bs=1 for parallel inferencing.")
    parser.add_argument("--soc", choices=["Duo", "A2"], default="A2", help="soc_version.")
    parser.add_argument(
        "--unet_data_dir",
        type=str,
        default='./unet_data.npy',
        help="save unet input for quant."
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="NPU device",
    )
    parser.add_argument(
        "--height",
        default=1024,
        type=int,
        help="image height",
    )
    parser.add_argument(
        "--width",
        default=1024,
        type=int,
        help="image width"
    )
    return parser.parse_args()

def trace_clip(sd_pipeline, batch_size, clip_pt_path, clip2_pt_path):
    encoder_model = sd_pipeline.text_encoder
    encoder_2_model = sd_pipeline.text_encoder_2
    max_position_embeddings = encoder_model.config.max_position_embeddings
    dummy_input = torch.ones([batch_size, max_position_embeddings], dtype=torch.int64)

    if not os.path.exists(clip_pt_path):
        clip_export = ClipExport(encoder_model)
        torch.jit.trace(clip_export, dummy_input).save(clip_pt_path)
    if not os.path.exists(clip2_pt_path):
        clip_export = ClipExport(encoder_2_model)
        torch.jit.trace(clip_export, dummy_input).save(clip2_pt_path)

def export_clip(sd_pipeline, args):
    print("Exporting the text encoder...")
    clip_path = os.path.join(args.output_dir, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o640)
    batch_size = args.batch_size
    clip_pt_path = os.path.join(clip_path, f"clip_bs{batch_size}.pt")
    clip2_pt_path = os.path.join(clip_path, f"clip2_bs{batch_size}.pt")
    clip1_compiled_path = os.path.join(clip_path, f"clip_bs{batch_size}_compile_quant_{args.height}x{args.width}.ts")
    clip2_compiled_path = os.path.join(clip_path, f"clip2_bs{batch_size}_compile_quant_{args.height}x{args.width}.ts")

    encoder_model = sd_pipeline.text_encoder
    max_position_embeddings = encoder_model.config.max_position_embeddings
    
    # trace
    trace_clip(sd_pipeline, batch_size, clip_pt_path, clip2_pt_path)

    # compile
    if not os.path.exists(clip1_compiled_path):
        model = torch.jit.load(clip_pt_path).eval()
        inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
        compile_clip(model, inputs, clip1_compiled_path, soc_version)
    if not os.path.exists(clip2_compiled_path):
        model = torch.jit.load(clip2_pt_path).eval()
        inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
        compile_clip(model, inputs, clip2_compiled_path, soc_version)

def export_vae(sd_pipeline, args):
    print("Exporting the image decoder...")
    vae_path = os.path.join(args.output_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    batch_size = args.batch_size
    height_size, width_size = args.height // 8, args.width // 8
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    vae_compiled_path = os.path.join(vae_path, f"vae_bs{batch_size}_compile_quant_{args.height}x{args.width}.ts")

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet
    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels

    # trace
    if not os.path.exists(vae_pt_path):
        dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32)
        vae_export = VaeExport(vae_model)
        torch.jit.trace(vae_export, dummy_input).save(vae_pt_path)

    # compile
    if not os.path.exists(vae_compiled_path):
        model = torch.jit.load(vae_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, height_size, width_size), dtype=mindietorch.dtype.FLOAT)]
        compile_vae(model, inputs, vae_compiled_path, soc_version)

def trace_ddim(sd_pipeline, args, ddim_pt_path):
    batch_size = args.batch_size * 2
    if not os.path.exists(ddim_pt_path):
        dummy_input = (
            torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.randn([batch_size // 2, 4, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
        )
        scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)
        scheduler.set_timesteps(args.steps, device="cpu")

        timesteps = scheduler.timesteps[:args.steps]
        alpha_prod_t_prev_cache = []
        for timestep in timesteps:
            prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            alpha_prod_t_prev = scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
            alpha_prod_t_prev_cache.append(alpha_prod_t_prev)

        new_ddim = NewScheduler(
            num_train_timesteps=scheduler.config.num_train_timesteps,
            num_inference_steps=scheduler.num_inference_steps,
            alphas_cumprod=scheduler.alphas_cumprod,
            guidance_scale=args.guidance_scale,
            alpha_prod_t_prev_cache=torch.tensor(alpha_prod_t_prev_cache)
        )
        new_ddim.eval()
        torch.jit.trace(new_ddim, dummy_input).save(ddim_pt_path)

def export_ddim(sd_pipeline, args):
    print("Exporting the ddim...")
    ddim_path = os.path.join(args.output_dir, "ddim")
    if not os.path.exists(ddim_path):
        os.makedirs(ddim_path, mode=0o744)
    batch_size = args.batch_size * 2
    height_size, width_size = args.height // 8, args.width // 8
    ddim_pt_path = os.path.join(ddim_path, f"ddim_bs{batch_size}.pt")
    scheduler_compiled_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_compile_quant_{args.height}x{args.width}.ts")

    unet_model = sd_pipeline.unet
    ddim_model = sd_pipeline.scheduler
    sample_size = unet_model.config.sample_size
    in_channels = 4

    # trace
    trace_ddim(sd_pipeline, args, ddim_pt_path)
    # compile
    if not os.path.exists(scheduler_compiled_path):
        model = torch.jit.load(ddim_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input((batch_size // 2, in_channels, height_size, width_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)
        ]
        compile_ddim(model, inputs, scheduler_compiled_path, soc_version)

def trace_ddim_parallel(sd_pipeline, args, ddim_pt_path):
    batch_size = args.batch_size
    if not os.path.exists(ddim_pt_path):
        dummy_input = (
            torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),
            torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
        )
        scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)
        scheduler.set_timesteps(args.steps, device="cpu")

        timesteps = scheduler.timesteps[:args.steps]
        alpha_prod_t_prev_cache = []
        for timestep in timesteps:
            prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            alpha_prod_t_prev = scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
            alpha_prod_t_prev_cache.append(alpha_prod_t_prev)

        new_ddim = Scheduler(
            num_train_timesteps=scheduler.config.num_train_timesteps,
            num_inference_steps=scheduler.num_inference_steps,
            alphas_cumprod=scheduler.alphas_cumprod,
            guidance_scale=args.guidance_scale,
            alpha_prod_t_prev_cache=torch.tensor(alpha_prod_t_prev_cache)
        )
        new_ddim.eval()
        torch.jit.trace(new_ddim, dummy_input).save(ddim_pt_path)

def export_ddim_parallel(sd_pipeline, args):
    print("Exporting the ddim...")
    ddim_path = os.path.join(args.output_dir, "ddim")
    if not os.path.exists(ddim_path):
        os.makedirs(ddim_path, mode=0o640)
    batch_size = args.batch_size
    height_size, width_size = args.height // 8, args.width // 8
    ddim_pt_path = os.path.join(ddim_path, f"ddim_bs{batch_size}.pt")
    scheduler_compiled_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_parallel_compile_quant_{args.height}x{args.width}.ts")

    in_channels = 4

    # trace
    trace_ddim_parallel(sd_pipeline, args, ddim_pt_path)
    # compile
    if not os.path.exists(scheduler_compiled_path):
        model = torch.jit.load(ddim_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)]
        compile_ddim(model, inputs, scheduler_compiled_path, soc_version)

def trace_quant_model(model, calib_datas, input_shape, pt_path):
    save_path = pt_path[:-3]
    quant_model = copy.deepcopy(model)
    export_model = copy.deepcopy(model)
    quant_config = QuantConfig(disable_names=[],
                               amp_num=0, input_shape=input_shape,
                               act_method=0, quant_mode=0, a_signed=True)
    calibrator = Calibrator(quant_model, quant_config, calib_data=calib_datas)
    calibrator.run()
    calibrator.export_param(os.path.join(save_path, 'quant_weights'))
    input_scale = np.load(os.path.join(save_path, 'quant_weights', 'input_scale.npy'), allow_pickle=True).item()
    input_offset = np.load(os.path.join(save_path, 'quant_weights', 'input_offset.npy'), allow_pickle=True).item()
    weight_scale = np.load(os.path.join(save_path, 'quant_weights', 'input_scale.npy'), allow_pickle=True).item()
    weight_offset = np.load(os.path.join(save_path, 'quant_weights', 'weight_scale.npy'), allow_pickle=True).item()
    quant_weight = np.load(os.path.join(save_path, 'quant_weights', 'quant_weight.npy'), allow_pickle=True).item()

    export_model = modify_model(export_model, input_scale, input_offset, weight_scale, weight_offset, quant_weight)
    torch.jit.trace(export_model, calib_datas[0]).save(pt_path)

def export_unet_cache(sd_pipeline, args, input_data):
    print("Exporting the image information creater...")
    unet_path = os.path.join(args.output_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    if input_data['parallel']:
        parallel = "parallel_"
        batch_size = args.batch_size
    else:
        parallel = ""
        batch_size = args.batch_size * 2
    height_size, width_size = args.height // 8, args.width // 8
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_0.pt")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_{parallel}compile_0_quant_{args.height}x{args.width}.ts")

    unet_model = copy.deepcopy(sd_pipeline.unet)
    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    if not os.path.exists(unet_pt_path):
        calib_datas = [list(input_data['cache'])]
        unet = UnetExport(unet_model)
        unet.eval()
        trace_quant_model(unet, calib_datas, [batch_size, in_channels, sample_size, sample_size], unet_pt_path)
    # compile
    if not os.path.exists(unet_compiled_path):
        model = torch.jit.load(unet_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, encoder_hidden_size_2),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)]
        compile_unet_cache(model, inputs, unet_compiled_path, soc_version)

def export_unet_skip(sd_pipeline, args, input_data):
    print("Exporting the image information creater...")
    unet_path = os.path.join(args.output_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    if input_data['parallel']:
        parallel = "parallel_"
        batch_size = args.batch_size
    else:
        parallel = ""
        batch_size = args.batch_size * 2
    height_size, width_size = args.height // 8, args.width // 8
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_1.pt")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_{parallel}compile_1_quant_{args.height}x{args.width}.ts")

    unet_model = copy.deepcopy(sd_pipeline.unet)
    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    if not os.path.exists(unet_pt_path):
        calib_datas = [list(input_data['skip'])]
        unet = UnetExport(unet_model)
        unet.eval()
        trace_quant_model(unet, calib_datas, [batch_size, in_channels, sample_size, sample_size], unet_pt_path)
    # compile
    if not os.path.exists(unet_compiled_path):
        model = torch.jit.load(unet_pt_path).eval()
        inputs = [
                mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                    dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                    dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, encoder_hidden_size_2),
                                    dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                mindietorch.Input((batch_size, 1280, math.ceil(height_size / 2), math.ceil(width_size / 2)),
                                  dtype=mindietorch.dtype.FLOAT)]
        compile_unet_skip(model, inputs, unet_compiled_path, soc_version)


def export_unet_init(sd_pipeline, args, input_data):
    print("Exporting the image information creater...")
    unet_path = os.path.join(args.output_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    batch_size = args.batch_size * 2
    height_size, width_size = args.height // 8, args.width // 8
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_quant_{args.height}x{args.width}.ts")

    unet_model = copy.deepcopy(sd_pipeline.unet)
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2
    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    if not os.path.exists(unet_pt_path):
        calib_datas = [list(input_data['no_cache'])]
        unet = UnetExportInit(unet_model)
        unet.eval()
        trace_quant_model(unet, calib_datas, [batch_size, in_channels, sample_size, sample_size], unet_pt_path)
    # compile
    if not os.path.exists(unet_compiled_path):
        model = torch.jit.load(unet_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, encoder_hidden_size_2),
                                dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT)]
        compile_unet_init(model, inputs, unet_compiled_path, soc_version)

def export(args):
    pipeline = StableDiffusionXLPipeline.from_pretrained(args.model).to('cpu')
    data = np.load(args.unet_data_dir, allow_pickle=True).item()
    print(data.keys())
    print(data['use_cache'])
    if 'use_cache' not in data or 'parallel' not in data:
        raise RuntimeError(f'invalid unet data file.')

    export_clip(pipeline, args)
    export_vae(pipeline, args)
    if data['use_cache']:
        export_unet_cache(pipeline, args, data)
        export_unet_skip(pipeline, args, data)
    else:
        export_unet_init(pipeline, args, data)
    if args.parallel:
        export_ddim_parallel(pipeline, args)
    else:
        export_ddim(pipeline, args)

def main():
    args = parse_arguments()
    mindietorch.set_device(args.device)
    export(args)
    print("Done.")
    mindietorch.finalize()

if __name__ == '__main__':
    args = parse_arguments()
    if args.soc == "Duo":
        soc_version = "Ascend310P3"
    elif args.soc == "A2":
        soc_version = "Ascend910B4"
    main()