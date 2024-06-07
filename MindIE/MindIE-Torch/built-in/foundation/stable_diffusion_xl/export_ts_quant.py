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

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from diffusers import StableDiffusionXLPipeline


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
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "-steps",
        "--steps",
        type=int,
        default=50,
        help="steps."
    )
    parser.add_argument(
        "-guid",
        "--guidance_scale",
        type=float,
        default=5.0,
        help="guidance_scale"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cache during inference."
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Export the unet of bs=1 for parallel inferencing.",
    )
    parser.add_argument(
        "--unet_data_dir",
        type=str,
        default='./unet_data.npy',
        help="save unet input for quant."
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
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        model_output = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_prod_t_prev_cache[step_index]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample


def export_ddim(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, steps: int, guidance_scale: float,
                batch_size: int) -> None:
    print("Exporting the ddim...")
    ddim_path = os.path.join(save_dir, "ddim")
    if not os.path.exists(ddim_path):
        os.makedirs(ddim_path, mode=0o744)

    ddim_pt_path = os.path.join(ddim_path, f"ddim{batch_size}.pt")
    if os.path.exists(ddim_pt_path):
        return

    ddim_model = sd_pipeline.scheduler

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


class Scheduler(torch.nn.Module):
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50, alphas_cumprod=None,
                 guidance_scale=5.0, alpha_prod_t_prev_cache=None):
        super(Scheduler, self).__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.alphas_cumprod = alphas_cumprod
        self.guidance_scale = guidance_scale
        self.alpha_prod_t_prev_cache = alpha_prod_t_prev_cache

    def forward(self, noise_pred_uncond: torch.FloatTensor, noise_pred_text: torch.FloatTensor, timestep: int,
                sample: torch.FloatTensor, step_index: int):
        model_output = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_prod_t_prev_cache[step_index]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample


def export_ddim_parallel(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, steps: int,
                         guidance_scale: float, batch_size: int) -> None:
    print("Exporting the ddim...")
    ddim_path = os.path.join(save_dir, "ddim")
    if not os.path.exists(ddim_path):
        os.makedirs(ddim_path, mode=0o640)
    ddim_pt_path = os.path.join(ddim_path, f"ddim{batch_size}.pt")
    if os.path.exists(ddim_pt_path):
        return

    ddim_model = sd_pipeline.scheduler
    dummy_input = (
        torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),  # 无条件噪声预测
        torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),  # 有条件噪声预测
        torch.ones([1], dtype=torch.int64),
        torch.randn([batch_size, 4, 128, 128], dtype=torch.float32),  # latent feature，1*4*64*64
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

    new_ddim = Scheduler(
        num_train_timesteps=scheduler.config.num_train_timesteps,
        num_inference_steps=scheduler.num_inference_steps,
        alphas_cumprod=scheduler.alphas_cumprod,
        guidance_scale=guidance_scale,
        alpha_prod_t_prev_cache=torch.tensor(alpha_prod_t_prev_cache)
    )

    new_ddim.eval()

    torch.jit.trace(new_ddim, dummy_input).save(ddim_pt_path)


class ClipExport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, output_hidden_states=True, return_dict=False):
        return self.clip_model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)


def export_clip(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the text encoder...")
    clip_path = os.path.join(save_dir, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o640)
    clip_pt_path = os.path.join(clip_path, f"clip_bs{batch_size}.pt")
    clip2_pt_path = os.path.join(clip_path, f"clip2_bs{batch_size}.pt")
    if os.path.exists(clip_pt_path) and os.path.exists(clip2_pt_path):
        return

    encoder_model = sd_pipeline.text_encoder
    encoder_2_model = sd_pipeline.text_encoder_2

    max_position_embeddings = encoder_model.config.max_position_embeddings
    print(f'max_position_embeddings: {max_position_embeddings}')

    dummy_input = torch.ones([batch_size, max_position_embeddings], dtype=torch.int64)

    if not os.path.exists(clip_pt_path):
        clip_export = ClipExport(encoder_model)
        torch.jit.trace(clip_export, dummy_input).save(clip_pt_path)
    if not os.path.exists(clip2_pt_path):
        clip_export = ClipExport(encoder_2_model)
        torch.jit.trace(clip_export, dummy_input).save(clip2_pt_path)


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


def export_unet_init(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int, input_data: dict) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    if os.path.exists(unet_pt_path):
        return

    unet_model = sd_pipeline.unet

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels

    calib_datas = [list(input_data['no_cache'])]
    unet = UnetExportInit(unet_model)
    unet.eval()

    trace_quant_model(unet, calib_datas, [batch_size, in_channels, sample_size, sample_size], unet_pt_path)


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


def export_unet(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int, if_skip: int,
                input_data: dict) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_{if_skip}.pt")
    if os.path.exists(unet_pt_path):
        return

    unet_model = sd_pipeline.unet

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels

    calib_datas = [list(input_data['skip'])] if if_skip else [list(input_data['cache'])]
    unet = UnetExport(unet_model)
    unet.eval()

    trace_quant_model(unet, calib_datas, [batch_size, in_channels, sample_size, sample_size], unet_pt_path)


def trace_quant_model(model, calib_datas, input_shape, pt_path):
    save_path = pt_path[:-3]
    export_model = copy.deepcopy(model)
    quant_config = QuantConfig(disable_names=[],
                               amp_num=0, input_shape=input_shape,
                               act_method=0, quant_mode=0, a_signed=True)
    calibrator = Calibrator(model, quant_config, calib_data=calib_datas)
    calibrator.run()
    calibrator.export_param(os.path.join(save_path, 'quant_weights'))
    input_scale = np.load(os.path.join(save_path, 'quant_weights', 'input_scale.npy'), allow_pickle=True).item()
    input_offset = np.load(os.path.join(save_path, 'quant_weights', 'input_offset.npy'), allow_pickle=True).item()
    weight_scale = np.load(os.path.join(save_path, 'quant_weights', 'input_scale.npy'), allow_pickle=True).item()
    weight_offset = np.load(os.path.join(save_path, 'quant_weights', 'weight_scale.npy'), allow_pickle=True).item()
    quant_weight = np.load(os.path.join(save_path, 'quant_weights', 'quant_weight.npy'), allow_pickle=True).item()

    export_model = modify_model(export_model, input_scale, input_offset, weight_scale, weight_offset, quant_weight)
    torch.jit.trace(export_model, calib_datas[0]).save(pt_path)


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        return self.vae_model.decoder(latents)


def export_vae(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    if os.path.exists(vae_pt_path):
        return

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels

    dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32)
    vae_export = VaeExport(vae_model)
    torch.jit.trace(vae_export, dummy_input).save(vae_pt_path)


if __name__ == '__main__':
    args = parse_arguments()
    pipeline = StableDiffusionXLPipeline.from_pretrained(args.model).to("cpu")

    data = np.load(args.unet_data_dir, allow_pickle=True).item()
    print(data.keys())
    print(data['use_cache'])
    if 'use_cache' not in data or 'parallel' not in data:
        raise RuntimeError(f'invalid unet data file.')

    export_clip(pipeline, args.output_dir, args.batch_size)
    export_vae(pipeline, args.output_dir, args.batch_size)
    if data['use_cache']:
        if data['parallel']:
            # 双卡并行带unetcache
            export_unet(pipeline, args.output_dir, args.batch_size, 0, data)
            export_unet(pipeline, args.output_dir, args.batch_size, 1, data)
        else:
            # 单卡带unetcache
            export_unet(pipeline, args.output_dir, args.batch_size * 2, 0, data)
            export_unet(pipeline, args.output_dir, args.batch_size * 2, 1, data)
    else:
        if data['parallel']:
            print("parallel without cache function is not currently supported.")
        else:
            # 单卡不带unetcache
            export_unet_init(pipeline, args.output_dir, args.batch_size * 2, data)

    if args.parallel:
        # 双卡
        export_ddim_parallel(pipeline, args.output_dir, args.steps, args.guidance_scale, batch_size)
    else:
        # 单卡
        export_ddim(pipeline, args.output_dir, args.steps, args.guidance_scale, batch_size * 2)

    print('succ')
