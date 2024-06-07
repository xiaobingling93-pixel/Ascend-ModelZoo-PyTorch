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
import mindietorch
from mindietorch import _enums
import math
from compile_model import compile_clip, compile_vae, compile_ddim,\
    compile_unet_cache, compile_unet_skip, compile_unet_init


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
    parser.add_argument("--soc", choices=["Duo", "A2"], default="A2", help="soc_version.", )
    parser.add_argument(
        "--flag",
        type=int,
        default=1,
        choices=[0, 1],
        help="0 is static; 1 is dynamic rankl.",
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="NPU device",
    )

    return parser.parse_args()


class NewScheduler(torch.nn.Module):
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50, alphas_cumprod=None,
                 guidance_scale=5.0, alpha_prod_t_prev_cache=None):
        super(NewScheduler, self).__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.alphas_cumprod = alphas_cumprod
        self.guidance_scale = guidance_scale
        self.alpha_prod_t_prev_cache = alpha_prod_t_prev_cache

    def forward(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, step_index: int):
        divide_batch = (model_output.shape[0]) // 2
        noise_pred_uncond = model_output[:divide_batch, ..., ..., ...]
        noise_pred_text = model_output[divide_batch:, ..., ..., ...]
        model_output = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_prod_t_prev_cache[step_index]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample


def trace_ddim(sd_pipeline, steps, guidance_scale, batch_size, ddim_pt_path):
    if not os.path.exists(ddim_pt_path):
        dummy_input = (
            torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.randn([batch_size // 2, 4, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
        )
        scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)
        scheduler.set_timesteps(steps, device="cpu")

        timesteps = scheduler.timesteps[:steps]
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
            guidance_scale=guidance_scale,
            alpha_prod_t_prev_cache=torch.tensor(alpha_prod_t_prev_cache)
        )

        new_ddim.eval()
        torch.jit.trace(new_ddim, dummy_input).save(ddim_pt_path)


def export_ddim(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, steps: int, guidance_scale: float,
                batch_size: int, flag: int) -> None:
    print("Exporting the ddim...")
    ddim_path = os.path.join(save_dir, "ddim")
    if not os.path.exists(ddim_path):
        os.makedirs(ddim_path, mode=0o744)

    ddim_pt_path = os.path.join(ddim_path, f"ddim_bs{batch_size}.pt")
    scheduler_compiled_static_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_compile_static.ts")
    scheduler_compiled_path = os.path.join(ddim_path, f"ddim_bs{batch_size}_compile.ts")

    unet_model = sd_pipeline.unet
    ddim_model = sd_pipeline.scheduler
    sample_size = unet_model.config.sample_size

    in_channels = 4
    # trace
    trace_ddim(sd_pipeline, steps, guidance_scale, batch_size, ddim_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(scheduler_compiled_static_path):
            model = torch.jit.load(ddim_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input((batch_size // 2,
                                         in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64)]
            compile_ddim(model, inputs, scheduler_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(scheduler_compiled_path):
            model = torch.jit.load(ddim_pt_path).eval()
            # 动态分档
            inputs = []
            inputs_gear_1 = [mindietorch.Input((batch_size, in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size // 2,
                                                in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64)]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((batch_size, in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size // 2,
                                                in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64)]
            inputs.append(inputs_gear_2)
            compile_ddim(model, inputs, scheduler_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


class ClipExport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, output_hidden_states=True, return_dict=False):
        return self.clip_model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)


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


def export_clip(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int, flag: int) -> None:
    print("Exporting the text encoder...")
    clip_path = os.path.join(save_dir, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o640)
    clip_pt_path = os.path.join(clip_path, f"clip_bs{batch_size}.pt")
    clip2_pt_path = os.path.join(clip_path, f"clip2_bs{batch_size}.pt")
    clip1_compiled_path = os.path.join(clip_path, f"clip_bs{batch_size}_compile.ts")
    clip2_compiled_path = os.path.join(clip_path, f"clip2_bs{batch_size}_compile.ts")

    encoder_model = sd_pipeline.text_encoder
    encoder_2_model = sd_pipeline.text_encoder_2

    max_position_embeddings = encoder_model.config.max_position_embeddings
    print(f'max_position_embeddings: {max_position_embeddings}')

    # trace
    trace_clip(sd_pipeline, batch_size, clip_pt_path, clip2_pt_path)

    # compile
    if flag == 0 or flag == 1:
        if not os.path.exists(clip1_compiled_path):
            model = torch.jit.load(clip_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
            compile_clip(model, inputs, clip1_compiled_path, soc_version)
        if not os.path.exists(clip2_compiled_path):
            model = torch.jit.load(clip2_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
            compile_clip(model, inputs, clip2_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


class UnetExportInit(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(
            self,
            sample,
            timestep,
            encoder_hidden_states,
            text_embeds,
            time_ids
    ):
        return self.unet_model(sample, timestep, encoder_hidden_states,
                               added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids})[0]


def trace_unet_init(sd_pipeline, batch_size, unet_pt_path):
    unet_model = sd_pipeline.unet
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2
    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    max_position_embeddings = encoder_model.config.max_position_embeddings
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size

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

        unet = UnetExportInit(unet_model)
        unet.eval()
        torch.jit.trace(unet, dummy_input).save(unet_pt_path)


def export_unet_init(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int, flag: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    compile_batch_size = batch_size * 2
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{compile_batch_size}_compile_static.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{compile_batch_size}_compile.ts")

    unet_model = sd_pipeline.unet
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    trace_unet_init(sd_pipeline, batch_size, unet_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(unet_compiled_static_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = [mindietorch.Input((compile_batch_size, in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input((compile_batch_size,
                                         max_position_embeddings,
                                         encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((compile_batch_size,
                                         encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((compile_batch_size, 6),
                                        dtype=mindietorch.dtype.FLOAT)]
            compile_unet_init(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((compile_batch_size, in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((compile_batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT)
                             ]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((compile_batch_size, in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((compile_batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT)
                             ]
            inputs.append(inputs_gear_2)
            compile_unet_init(model, inputs, unet_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


class UnetExport(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(
            self,
            sample,
            timestep,
            encoder_hidden_states,
            text_embeds,
            time_ids,
            if_skip,
            inputCache=None
    ):
        if if_skip:
            return self.unet_model(sample, timestep, encoder_hidden_states,
                                   added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                                   if_skip=if_skip, inputCache=inputCache)[0]
        else:
            return self.unet_model(sample, timestep, encoder_hidden_states,
                                   added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                                   if_skip=if_skip)


def export_unet_skip(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int, flag: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    compile_batch_size = batch_size * 2

    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_1.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{compile_batch_size}_compile_1_static.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{compile_batch_size}_compile_1_haveshape.ts")
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
            torch.ones([batch_size, 1280, math.ceil(sample_size / 2), math.ceil(sample_size / 2)], dtype=torch.float32),
        )
        unet = UnetExport(unet_model)
        unet.eval()
        torch.jit.trace(unet, dummy_input).save(unet_pt_path)
    # compile
    if flag == 0:
        if not os.path.exists(unet_compiled_static_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = [mindietorch.Input((compile_batch_size, in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input((compile_batch_size,
                                         max_position_embeddings,
                                         encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((compile_batch_size,
                                         encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((compile_batch_size, 6),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input(
                          (compile_batch_size, 1280, math.ceil(sample_size / 2),
                           math.ceil(sample_size / 2)),
                          dtype=mindietorch.dtype.FLOAT)]
            compile_unet_skip(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((compile_batch_size, in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((compile_batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input(
                                 (compile_batch_size, 1280, math.ceil(1024 // 8 / 2),
                                  math.ceil(1024 // 8 / 2)),
                                 dtype=mindietorch.dtype.FLOAT),
                             ]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((compile_batch_size, in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((compile_batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input(
                                 (compile_batch_size, 1280, math.ceil(512 // 8 / 2),
                                  math.ceil(512 // 8 / 2)),
                                 dtype=mindietorch.dtype.FLOAT),
                             ]
            inputs.append(inputs_gear_2)
            compile_unet_skip(model, inputs, unet_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


def export_unet_cache(sd_pipeline: StableDiffusionXLPipeline, save_dir: str,
                      batch_size: int, flag: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_0.pt")
    compile_batch_size = batch_size * 2
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{compile_batch_size}_compile_0_static.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{compile_batch_size}_compile_0.ts")
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
        unet = UnetExport(unet_model)
        unet.eval()

        torch.jit.trace(unet, dummy_input).save(unet_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(unet_compiled_static_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = [mindietorch.Input((compile_batch_size, in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input((compile_batch_size,
                                         max_position_embeddings,
                                         encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((compile_batch_size,
                                         encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((compile_batch_size, 6),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64)]
            compile_unet_cache(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((compile_batch_size,
                                                in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((compile_batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             ]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((compile_batch_size,
                                                in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((compile_batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((compile_batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             ]
            inputs.append(inputs_gear_2)
            compile_unet_cache(model, inputs, unet_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        return self.vae_model.decoder(latents)


def export_vae(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int, flag: int) -> None:
    print("Exporting the image decoder...")

    compile_batch_size = batch_size * 2

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    vae_compiled_static_path = os.path.join(vae_path, f"vae_bs{compile_batch_size}_compile_static.ts")
    vae_compiled_path = os.path.join(vae_path, f"vae_bs{compile_batch_size}_compile.ts")

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
        if not os.path.exists(vae_compiled_static_path):
            model = torch.jit.load(vae_pt_path).eval()
            # 静态shape
            inputs = [
                mindietorch.Input((compile_batch_size, in_channels, sample_size, sample_size),
                                  dtype=mindietorch.dtype.FLOAT)]
            compile_vae(model, inputs, vae_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(vae_compiled_path):
            # 动态分档
            model = torch.jit.load(vae_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((compile_batch_size, in_channels,
                                                1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((compile_batch_size, in_channels,
                                                512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_2)

            compile_vae(model, inputs, vae_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


def export(model_path: str, save_dir: str, batch_size: int, steps: int, guidance_scale: float, use_cache: bool,
           flag: int) -> None:
    pipeline = StableDiffusionXLPipeline.from_pretrained(model_path).to("cpu")

    export_clip(pipeline, save_dir, batch_size, flag)
    export_vae(pipeline, save_dir, batch_size, flag)

    if use_cache:
        # 单卡带unetcache
        export_unet_cache(pipeline, save_dir, batch_size * 2, flag)
        export_unet_skip(pipeline, save_dir, batch_size * 2, flag)
    else:
        # 单卡不带unetcache
        export_unet_init(pipeline, save_dir, batch_size * 2, flag)

    # 单卡
    export_ddim(pipeline, save_dir, steps, guidance_scale, batch_size * 2, flag)


def main():
    args = parse_arguments()
    export(args.model,
           args.output_dir,
           args.batch_size,
           args.steps,
           args.guidance_scale,
           args.use_cache,
           args.flag)
    print("Done.")
    mindietorch.finalize()


if __name__ == "__main__":
    min_batch, max_batch = 1, 32
    min_height, max_height = 512 // 8, 1024 // 8
    min_width, max_width = 512 // 8, 1664 // 8
    args = parse_arguments()
    mindietorch.set_device(args.device)
    if args.soc == "Duo":
        soc_version = "Ascend310P3"
    elif args.soc == "A2":
        soc_version = "Ascend910B4"
    else:
        print("unsupport soc_version, please check!")
    main()