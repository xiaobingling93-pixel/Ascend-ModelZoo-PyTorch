# Copyright 2024 Huawei Technologies Co., Ltd
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
import argparse
from argparse import Namespace

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from diffusers import StableDiffusionXLPipeline
import math
from compile_model import *
import mindietorch

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
        "--flag",
        choices=[0, 1, 2],
        default=0,
        type=int,
        help="0 is static; 1 is dynami dims; 2 is dynamic range.",
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
    flag, batch_size = args.flag, args.batch_size
    clip_pt_path = os.path.join(clip_path, f"clip_bs{batch_size}.pt")
    clip2_pt_path = os.path.join(clip_path, f"clip2_bs{batch_size}.pt")
    clip1_compiled_static_path = os.path.join(clip_path, f"clip_bs{batch_size}_compile_static_{args.height}x{args.width}.ts")
    clip2_compiled_static_path = os.path.join(clip_path, f"clip2_bs{batch_size}_compile_static_{args.height}x{args.width}.ts")
    clip1_compiled_path = os.path.join(clip_path, f"clip_bs{batch_size}_compile.ts")
    clip2_compiled_path = os.path.join(clip_path, f"clip2_bs{batch_size}_compile.ts")
    clip1_compiled_dynamic_path = os.path.join(clip_path, f"clip_compile_dynamic.ts")
    clip2_compiled_dynamic_path = os.path.join(clip_path, f"clip2_compile_dynamic.ts")

    encoder_model = sd_pipeline.text_encoder
    max_position_embeddings = encoder_model.config.max_position_embeddings
    
    # trace
    trace_clip(sd_pipeline, batch_size, clip_pt_path, clip2_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(clip1_compiled_static_path):
            model = torch.jit.load(clip_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
            compile_clip(model, inputs, clip1_compiled_static_path, soc_version)
        if not os.path.exists(clip2_compiled_static_path):
            model = torch.jit.load(clip2_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
            compile_clip(model, inputs, clip2_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(clip1_compiled_path):
            model = torch.jit.load(clip_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
            compile_clip(model, inputs, clip1_compiled_path, soc_version)
        if not os.path.exists(clip2_compiled_path):
            model = torch.jit.load(clip2_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
            compile_clip(model, inputs, clip2_compiled_path, soc_version)
    elif flag == 2:
        min_shape = (min_batch, max_position_embeddings)
        max_shape = (max_batch, max_position_embeddings)
        if not os.path.exists(clip1_compiled_dynamic_path):
            inputs = []
            inputs.append(mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=mindietorch.dtype.INT64))
            model = torch.jit.load(clip_pt_path).eval()
            compile_clip(model, inputs, clip1_compiled_dynamic_path, soc_version)
        if not os.path.exists(clip2_compiled_dynamic_path):
            inputs = []
            inputs.append(mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=mindietorch.dtype.INT64))
            model = torch.jit.load(clip2_pt_path).eval()
            compile_clip(model, inputs, clip2_compiled_dynamic_path, soc_version)

def export_vae(sd_pipeline, args):
    print("Exporting the image decoder...")
    vae_path = os.path.join(args.output_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    flag, batch_size = args.flag, args.batch_size
    height_size, width_size = args.height // 8, args.width // 8
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    vae_compiled_static_path = os.path.join(vae_path, f"vae_bs{batch_size}_compile_static_{args.height}x{args.width}.ts")
    vae_compiled_path = os.path.join(vae_path, f"vae_bs{batch_size}_compile.ts")
    vae_compiled_dynamic_path = os.path.join(vae_path, f"vae_compile_dynamic.ts")

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
    if flag == 0:
        # 静态
        if not os.path.exists(vae_compiled_static_path):
            model = torch.jit.load(vae_pt_path).eval()
            inputs = [
                mindietorch.Input((batch_size, in_channels, height_size, width_size), dtype=mindietorch.dtype.FLOAT)]
            compile_vae(model, inputs, vae_compiled_static_path, soc_version)
    elif flag == 1:
        # 动态dims
        if not os.path.exists(vae_compiled_path):
            model = torch.jit.load(vae_pt_path).eval()
            inputs = []
            for i in range(len(heights)):
                inputs_gear = [
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8), dtype=mindietorch.dtype.FLOAT)]
                inputs.append(inputs_gear)
            compile_vae(model, inputs, vae_compiled_path, soc_version)
    elif flag == 2:
        # 动态shape
        if not os.path.exists(vae_compiled_dynamic_path):
            model = torch.jit.load(vae_pt_path).eval()
            min_shape = (min_batch, in_channels, min_height, min_width)
            max_shape = (max_batch, in_channels, max_height, max_width)
            inputs = [mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=mindietorch.dtype.FLOAT)]
            compile_vae(model, inputs, vae_compiled_dynamic_path, soc_version)

def export_unet_init(sd_pipeline, args):
    print("Exporting the image information creater...")
    unet_path = os.path.join(args.output_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    flag, batch_size = args.flag, args.batch_size * 2
    height_size, width_size = args.height // 8, args.width // 8
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_static_{args.height}x{args.width}.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile.ts")
    unet_compiled_dynamic_path = os.path.join(unet_path, f"unet_compile_dynamic.ts")

    unet_model = sd_pipeline.unet
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    if not os.path.exists(unet_pt_path):
        dummy_input = (
            torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.ones(
                [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
            ),
            torch.ones([batch_size, encoder_hidden_size_2], dtype=torch.float32),
            torch.ones([batch_size, 6], dtype=torch.float32)
        )
        unet = UnetExportInit(unet_model).eval()
        torch.jit.trace(unet, dummy_input).save(unet_pt_path)
    
    # compile
    if flag == 0:
        # 静态
        if not os.path.exists(unet_compiled_static_path):
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
            compile_unet_init(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            for i in range(len(heights)):
                inputs_gear = [
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size, encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT)]
                inputs.append(inputs_gear)
            compile_unet_init(model, inputs, unet_compiled_path, soc_version)
    elif flag == 2:
        if not os.path.exists(unet_compiled_dynamic_path):
            model = torch.jit.load(unet_pt_path).eval()
            min_shape_1 = (min_batch * 2, in_channels, min_height, min_width)
            max_shape_1 = (max_batch * 2, in_channels, max_height, max_width)
            min_shape_2, max_shape_2 = (1,), (1,)
            min_shape_3 = (min_batch * 2, max_position_embeddings, encoder_hidden_size)
            max_shape_3 = (max_batch * 2, max_position_embeddings, encoder_hidden_size)
            min_shape_4 = (min_batch * 2, encoder_hidden_size_2)
            max_shape_4 = (max_batch * 2, encoder_hidden_size_2)
            min_shape_5 = (min_batch * 2, 6)
            max_shape_5 = (max_batch * 2, 6)
            inputs = [
                mindietorch.Input(min_shape=min_shape_1, max_shape=max_shape_1, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_2, max_shape=max_shape_2, dtype=mindietorch.dtype.INT64),
                mindietorch.Input(min_shape=min_shape_3, max_shape=max_shape_3, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_4, max_shape=max_shape_4, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_5, max_shape=max_shape_5, dtype=mindietorch.dtype.FLOAT),
            ]
            compile_unet_init(model, inputs, unet_compiled_dynamic_path, soc_version)

def export_unet_cache(sd_pipeline, args):
    print("Exporting the image information creater...")
    unet_path = os.path.join(args.output_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    if args.parallel:
        parallel = "parallel_"
        batch_size = args.batch_size
    else:
        parallel = ""
        batch_size = args.batch_size * 2
    flag = args.flag
    height_size, width_size = args.height // 8, args.width // 8
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_0.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{batch_size}_{parallel}compile_0_static_{args.height}x{args.width}.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_{parallel}compile_0.ts")
    unet_compiled_dynamic_path = os.path.join(unet_path, f"unet_{parallel}compile_0_dynamic.ts")
    
    unet_model = sd_pipeline.unet
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    if not os.path.exists(unet_pt_path):
        dummy_input = (
                torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
                torch.ones([1], dtype=torch.int64),
                torch.ones(
                    [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
                ),
                torch.ones([batch_size, encoder_hidden_size_2], dtype=torch.float32),
                torch.ones([batch_size, 6], dtype=torch.float32),
                torch.zeros([1], dtype=torch.int64),
            )
        unet = UnetExport(unet_model).eval()
        torch.jit.trace(unet, dummy_input).save(unet_pt_path)

    # compile
    if flag == 0:
        # 静态
        if not os.path.exists(unet_compiled_static_path):
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
            compile_unet_cache(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        # 动态dims
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            for i in range(len(heights)):
                inputs_gear = [
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size, encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)]
                inputs.append(inputs_gear)
            compile_unet_cache(model, inputs, unet_compiled_path, soc_version)
    elif flag == 2:
        if not os.path.exists(unet_compiled_dynamic_path):
            model = torch.jit.load(unet_pt_path).eval()
            if args.parallel:
                min_batch_temp, max_batch_temp = min_batch, max_batch
            else:
                min_batch_temp, max_batch_temp = min_batch * 2, max_batch * 2
            min_shape_1 = (min_batch_temp, in_channels, min_height, min_width)
            max_shape_1 = (max_batch_temp, in_channels, max_height, max_width)
            min_shape_2, max_shape_2 = (1,), (1,)
            min_shape_3 = (min_batch_temp, max_position_embeddings, encoder_hidden_size)
            max_shape_3 = (max_batch_temp, max_position_embeddings, encoder_hidden_size)
            min_shape_4 = (min_batch_temp, encoder_hidden_size_2)
            max_shape_4 = (max_batch_temp, encoder_hidden_size_2)
            min_shape_5 = (min_batch_temp, 6)
            max_shape_5 = (max_batch_temp, 6)
            min_shape_6, max_shape_6 = (1,), (1,)
            inputs = [
                mindietorch.Input(min_shape=min_shape_1, max_shape=max_shape_1, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_2, max_shape=max_shape_2, dtype=mindietorch.dtype.INT64),
                mindietorch.Input(min_shape=min_shape_3, max_shape=max_shape_3, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_4, max_shape=max_shape_4, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_5, max_shape=max_shape_5, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_6, max_shape=max_shape_6, dtype=mindietorch.dtype.INT64)
            ]
            compile_unet_cache(model, inputs, unet_compiled_dynamic_path, soc_version)

def export_unet_skip(sd_pipeline, args):
    print("Exporting the image information creater...")
    unet_path = os.path.join(args.output_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    if args.parallel:
        parallel = "parallel_"
        batch_size = args.batch_size
    else:
        parallel = ""
        batch_size = args.batch_size * 2
    flag = args.flag
    height_size, width_size = args.height // 8, args.width // 8
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_1.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{batch_size}_{parallel}compile_1_static_{args.height}x{args.width}.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_{parallel}compile_1.ts")
    unet_compiled_dynamic_path = os.path.join(unet_path, f"unet_{parallel}compile_1_dynamic.ts")
    
    unet_model = sd_pipeline.unet
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    if not os.path.exists(unet_pt_path):
        dummy_input = (
                torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
                torch.ones([1], dtype=torch.int64),
                torch.ones(
                    [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
                ),
                torch.ones([batch_size, encoder_hidden_size_2], dtype=torch.float32),
                torch.ones([batch_size, 6], dtype=torch.float32),
                torch.ones([1], dtype=torch.int64),
                torch.ones([batch_size, 1280, math.ceil(sample_size / 2), math.ceil(sample_size / 2)],
                           dtype=torch.float32)
            )
        unet = UnetExport(unet_model).eval()
        torch.jit.trace(unet, dummy_input).save(unet_pt_path)
    
    # compile
    if flag == 0:
        # 静态
        if not os.path.exists(unet_compiled_static_path):
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
            compile_unet_skip(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        # 动态dims
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            for i in range(len(heights)):
                inputs_gear = [
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size, encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size, 1280, math.ceil(heights[i] // 8 / 2), math.ceil(widths[i] // 8 / 2)),
                                    dtype=mindietorch.dtype.FLOAT)]
                inputs.append(inputs_gear)
            compile_unet_skip(model, inputs, unet_compiled_path, soc_version)
    elif flag == 2:
        if not os.path.exists(unet_compiled_dynamic_path):
            model = torch.jit.load(unet_pt_path).eval()
            if args.parallel:
                min_batch_temp, max_batch_temp = min_batch, max_batch
            else:
                min_batch_temp, max_batch_temp = min_batch * 2, max_batch * 2
            min_shape_1 = (min_batch_temp, in_channels, min_height, min_width)
            max_shape_1 = (max_batch_temp, in_channels, max_height, max_width)
            min_shape_2, max_shape_2 = (1,), (1,)
            min_shape_3 = (min_batch_temp, max_position_embeddings, encoder_hidden_size)
            max_shape_3 = (max_batch_temp, max_position_embeddings, encoder_hidden_size)
            min_shape_4 = (min_batch_temp, encoder_hidden_size_2)
            max_shape_4 = (max_batch_temp, encoder_hidden_size_2)
            min_shape_5 = (min_batch_temp, 6)
            max_shape_5 = (max_batch_temp, 6)
            min_shape_6, max_shape_6 = (1,), (1,)
            min_shape_7 = (min_batch_temp, 1280, math.ceil(min_height / 2), math.ceil(min_width / 2))
            max_shape_7 = (max_batch_temp, 1280, math.ceil(max_height / 2), math.ceil(max_width / 2))
            inputs = [
                mindietorch.Input(min_shape=min_shape_1, max_shape=max_shape_1, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_2, max_shape=max_shape_2, dtype=mindietorch.dtype.INT64),
                mindietorch.Input(min_shape=min_shape_3, max_shape=max_shape_3, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_4, max_shape=max_shape_4, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_5, max_shape=max_shape_5, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_6, max_shape=max_shape_6, dtype=mindietorch.dtype.INT64),
                mindietorch.Input(min_shape=min_shape_7, max_shape=max_shape_7, dtype=mindietorch.dtype.FLOAT)
            ]
            compile_unet_skip(model, inputs, unet_compiled_dynamic_path, soc_version)

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
    flag, batch_size = args.flag, args.batch_size * 2
    height_size, width_size = args.height // 8, args.width // 8
    ddim_pt_path = os.path.join(ddim_path, f"ddim_bs{batch_size}.pt")
    scheduler_compiled_static_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_compile_static_{args.height}x{args.width}.ts")
    scheduler_compiled_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_compile.ts")
    scheduler_compiled_dynamic_path = os.path.join(ddim_path, f"ddim_compile_dynamic.ts")

    in_channels = 4

    # trace
    trace_ddim(sd_pipeline, args, ddim_pt_path)

    # compile
    if flag == 0:
        # 静态
        if not os.path.exists(scheduler_compiled_static_path):
            model = torch.jit.load(ddim_pt_path).eval()
            inputs = [
                mindietorch.Input((batch_size, in_channels, height_size, width_size),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                mindietorch.Input((batch_size // 2, in_channels, height_size, width_size),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)
            ]
            compile_ddim(model, inputs, scheduler_compiled_static_path, soc_version)
    elif flag == 1:
        # 动态dims
        if not os.path.exists(scheduler_compiled_path):
            model = torch.jit.load(ddim_pt_path).eval()
            inputs = []
            for i in range(len(heights)):
                inputs_gear = [
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8),
                                    dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size // 2, in_channels, heights[i] // 8, widths[i] // 8),
                                    dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)]
                inputs.append(inputs_gear)
            compile_ddim(model, inputs, scheduler_compiled_path, soc_version)
    elif flag == 2:
        if not os.path.exists(scheduler_compiled_dynamic_path):
            model = torch.jit.load(ddim_pt_path).eval()
            min_shape_1 = (min_batch * 2, in_channels, min_height, min_width)
            max_shape_1 = (max_batch * 2, in_channels, max_height, max_width)
            min_shape_2, max_shape_2 = (1,), (1,)
            min_shape_3 = (min_batch, in_channels, min_height, min_width)
            max_shape_3 = (max_batch, in_channels, max_height, max_width)
            min_shape_4, max_shape_4 = (1,), (1,)
            inputs = [
                mindietorch.Input(min_shape=min_shape_1, max_shape=max_shape_1, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_2, max_shape=max_shape_2, dtype=mindietorch.dtype.INT64),
                mindietorch.Input(min_shape=min_shape_3, max_shape=max_shape_3, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_4, max_shape=max_shape_4, dtype=mindietorch.dtype.INT64)]
            compile_ddim(model, inputs, scheduler_compiled_dynamic_path, soc_version)

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
        os.makedirs(ddim_path, mode=0o744)
    flag, batch_size = args.flag, args.batch_size
    height_size, width_size = args.height // 8, args.width // 8
    ddim_pt_path = os.path.join(ddim_path, f"ddim_bs{batch_size}.pt")
    scheduler_compiled_static_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_parallel_compile_static_{args.height}x{args.width}.ts")
    scheduler_compiled_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_parallel_compile.ts")
    scheduler_compiled_dynamic_path = os.path.join(ddim_path, f"ddim_parallel_compile_dynamic.ts")

    in_channels = 4

    # trace
    trace_ddim_parallel(sd_pipeline, args, ddim_pt_path)

    # compile
    if flag == 0:
        # 静态
        if not os.path.exists(scheduler_compiled_static_path):
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
            compile_ddim(model, inputs, scheduler_compiled_static_path, soc_version)
    elif flag == 1:
        # 动态dims
        if not os.path.exists(scheduler_compiled_path):
            model = torch.jit.load(ddim_pt_path).eval()
            inputs = []
            for i in range(len(heights)):
                inputs_gear = [
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8),
                                    dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8),
                                    dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size, in_channels, heights[i] // 8, widths[i] // 8),
                                    dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)]
                inputs.append(inputs_gear)
            compile_ddim(model, inputs, scheduler_compiled_path, soc_version)
    elif flag == 2:
        if not os.path.exists(scheduler_compiled_dynamic_path):
            model = torch.jit.load(ddim_pt_path).eval()
            min_shape_1 = (min_batch, in_channels, min_height, min_width)
            max_shape_1 = (max_batch, in_channels, max_height, max_width)
            min_shape_2 = (min_batch, in_channels, min_height, min_width)
            max_shape_2 = (max_batch, in_channels, max_height, max_width)
            min_shape_3, max_shape_3 = (1,), (1,)
            min_shape_4 = (min_batch, in_channels, min_height, min_width)
            max_shape_4 = (max_batch, in_channels, max_height, max_width)
            min_shape_5, max_shape_5 = (1,), (1,)
            inputs = [
                mindietorch.Input(min_shape=min_shape_1, max_shape=max_shape_1, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_2, max_shape=max_shape_2, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_3, max_shape=max_shape_3, dtype=mindietorch.dtype.INT64),
                mindietorch.Input(min_shape=min_shape_4, max_shape=max_shape_4, dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input(min_shape=min_shape_5, max_shape=max_shape_5, dtype=mindietorch.dtype.INT64)]
            compile_ddim(model, inputs, scheduler_compiled_dynamic_path, soc_version)

def export(args):
    pipeline = StableDiffusionXLPipeline.from_pretrained(args.model).to('cpu')

    export_clip(pipeline, args)
    export_vae(pipeline, args)
    if args.use_cache:
        export_unet_cache(pipeline, args)
        export_unet_skip(pipeline, args)
    else:
        export_unet_init(pipeline, args)
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

if __name__ == "__main__":
    # 动态shape支持的分辨率
    min_batch, max_batch = 1, 32
    min_height, max_height = 512 // 8, 1024 // 8
    min_width, max_width = 512 // 8, 1664 // 8
    # 动态分档支持的分辨率
    heights = [1024, 512, 936, 768, 576]
    widths = [1024, 512, 1664, 1360, 1024]
    
    args = parse_arguments()
    if args.soc == "Duo":
        soc_version = "Ascend310P3"
    elif args.soc == "A2":
        soc_version = "Ascend910B4"
    main()