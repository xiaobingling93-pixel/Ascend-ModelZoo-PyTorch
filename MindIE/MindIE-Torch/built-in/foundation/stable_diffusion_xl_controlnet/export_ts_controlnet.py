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
import mindietorch
import torch.nn as nn

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
import torch
from compile_model import compile_clip, compile_vae, compile_unet, compile_control


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
        "-control",
        "--controlnet_model",
        type=str,
        default="./controlnet-canny-sdxl-1.0",
        help="Path or name of the pre-trained controlnet model.",
    )
    parser.add_argument(
        "-vae",
        "--vae_model",
        type=str,
        default="./sdxl-vae-fp16-fix",
        help="Path or name of the pre-trained vae model.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "-cond_scale",
        "--conditioning_scale",
        type=float,
        default=0.5,
        help="conditioning_scale"
    )
    parser.add_argument(
        "--flag",
        type=int,
        default=1,
        choices=[0, 1],
        help="0 is static; 1 is dynamic rankl.",
    )
    parser.add_argument(
        "--soc",
        choices=["A2"],
        default="A2",
        help="soc_version.",
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="NPU device",
    )

    return parser.parse_args()


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


def export_clip(sd_pipeline: StableDiffusionXLControlNetPipeline, save_dir: str,
                batch_size: int, flag: int,
                soc_version: str) -> None:
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
            latent_model_input,
            timestep,
            encoder_hidden_states,
            down_block_res_samples0,
            down_block_res_samples1,
            down_block_res_samples2,
            down_block_res_samples3,
            down_block_res_samples4,
            down_block_res_samples5,
            down_block_res_samples6,
            down_block_res_samples7,
            down_block_res_samples8,
            mid_block_res_sample,
            text_embeds,
            time_ids
    ):
        return self.unet_model(latent_model_input, timestep,
                               encoder_hidden_states=encoder_hidden_states,
                               down_block_additional_residuals=[down_block_res_samples0, down_block_res_samples1,
                                                                down_block_res_samples2,
                                                                down_block_res_samples3, down_block_res_samples4,
                                                                down_block_res_samples5, down_block_res_samples6,
                                                                down_block_res_samples7, down_block_res_samples8],
                               mid_block_additional_residual=mid_block_res_sample,
                               added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids})[0]


def export_unet(sd_pipeline: StableDiffusionXLControlNetPipeline, save_dir: str, batch_size: int, flag,
                soc_version: str) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_static.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile.ts")

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
            torch.ones([1], dtype=torch.float32),
            torch.ones(
                [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
            ),
            torch.ones([2, 320, 128, 128], dtype=torch.float32),
            torch.ones([2, 320, 128, 128], dtype=torch.float32),
            torch.ones([2, 320, 128, 128], dtype=torch.float32),
            torch.ones([2, 320, 64, 64], dtype=torch.float32),
            torch.ones([2, 640, 64, 64], dtype=torch.float32),
            torch.ones([2, 640, 64, 64], dtype=torch.float32),
            torch.ones([2, 640, 32, 32], dtype=torch.float32),
            torch.ones([2, 1280, 32, 32], dtype=torch.float32),
            torch.ones([2, 1280, 32, 32], dtype=torch.float32),
            torch.ones([2, 1280, 32, 32], dtype=torch.float32),
            torch.ones([batch_size, encoder_hidden_size_2], dtype=torch.float32),
            torch.ones([batch_size, 6], dtype=torch.float32)
        )

        unet = UnetExportInit(unet_model)
        unet.eval()

        torch.jit.trace(unet, dummy_input).save(unet_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(unet_compiled_static_path):
            model = torch.jit.load(unet_pt_path).eval()
            # 静态shape
            inputs = [
                mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 128, 128), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 128, 128), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 128, 128), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, encoder_hidden_size_2), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT)]
            compile_unet(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            # 动态分档
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            inputs_gear_1 = [
                mindietorch.Input((batch_size, in_channels, 1024 // 8, 1024 // 8,),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 128, 128), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 128, 128), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 128, 128), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, encoder_hidden_size_2), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [
                mindietorch.Input((batch_size, in_channels, 512 // 8, 512 // 8,),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
                mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                  dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 64, 64), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 320, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 32, 32), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 640, 16, 16), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 16, 16), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 16, 16), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((2, 1280, 16, 16), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, encoder_hidden_size_2), dtype=mindietorch.dtype.FLOAT),
                mindietorch.Input((batch_size, 6), dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_2)

            compile_unet(model, inputs, unet_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        return self.vae_model.decoder(latents)


def export_vae(sd_pipeline: StableDiffusionXLControlNetPipeline, save_dir: str, batch_size: int, flag: int,
               vaepath: str, soc_version: str) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    vae_compiled_static_path = os.path.join(vae_path, f"vae_bs{batch_size}_compile_static.ts")
    vae_compiled_path = os.path.join(vae_path, f"vae_bs{batch_size}_compile.ts")

    vae_model = AutoencoderKL.from_pretrained(vaepath)
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
                mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                                  dtype=mindietorch.dtype.FLOAT)]
            compile_vae(model, inputs, vae_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(vae_compiled_path):
            # 动态分档
            model = torch.jit.load(vae_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((batch_size, in_channels,
                                                1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((batch_size, in_channels,
                                                512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_2)

            compile_vae(model, inputs, vae_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


class ControlNetExport(torch.nn.Module):
    def __init__(self, controlnet, conditioning_scale):
        super().__init__()
        self.controlnet = controlnet
        self.conditioning_scale = conditioning_scale  # 0.5

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond, text_embeds, time_ids):
        return self.controlnet(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
                               controlnet_cond=controlnet_cond, conditioning_scale=self.conditioning_scale,
                               guess_mode=False,
                               added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                               return_dict=False)


def export_control(model, save_path, controlnet_path, conditioning_scale, flag, soc_version: str,
                   batch_size: int):
    control_path = os.path.join(save_path, "control")
    if not os.path.exists(control_path):
        os.makedirs(control_path, mode=0o744)
    control_pt_path = os.path.join(control_path, "control_bs{batch_size}.pt")
    control_compiled_static_path = os.path.join(control_path, f"control_bs{batch_size}_compile_static.ts")
    control_compiled_path = os.path.join(control_path, f"control_bs{batch_size}_compile.ts")
    controlnet = ControlNetModel.from_pretrained(controlnet_path)

    # trace
    if not os.path.exists(control_pt_path):
        dummy_input = (
            torch.ones([2, 4, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.float32),
            torch.ones([2, 77, 2048], dtype=torch.float32),
            torch.ones([2, 3, 1024, 1024], dtype=torch.float32),
            torch.ones([2, 1280], dtype=torch.float32),
            torch.ones([2, 6], dtype=torch.float32),
        )
        model_export = ControlNetExport(controlnet, conditioning_scale).eval()
        torch.jit.trace(model_export, dummy_input).save(control_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(control_compiled_static_path):
            model = torch.jit.load(control_pt_path).eval()
            # 静态shape
            inputs = [mindietorch.Input(([2, 4, 128, 128]), dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,), dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input(([2, 77, 2048]), dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input([2, 3, 1024, 1024], dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input([2, 1280], dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input([2, 6], dtype=mindietorch.dtype.FLOAT)]
            compile_control(model, inputs, control_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(control_compiled_path):
            # 动态分档
            model = torch.jit.load(control_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input(([2, 4, 128, 128]), dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,), dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input(([2, 77, 2048]), dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input([2, 3, 1024, 1024], dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input([2, 1280], dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input([2, 6], dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input(([2, 4, 64, 64]), dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,), dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input(([2, 77, 2048]), dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input([2, 3, 512, 512], dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input([2, 1280], dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input([2, 6], dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_2)

            compile_control(model, inputs, control_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


def export(model_path: str, controlnet_path: str, vae_path: str, save_dir: str, batch_size: int,
           conditioning_scale: float, flag: int, soc: str) -> None:
    if soc == "A2":
        soc_version = "Ascend910B4"
    else:
        print("unsupport soc_version, please check!")
        return

    controlnet = ControlNetModel.from_pretrained(controlnet_path)
    vae = AutoencoderKL.from_pretrained(vae_path)

    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(model_path,
                                                                   controlnet=controlnet, vae=vae).to("cpu")

    export_clip(pipeline, save_dir, batch_size, flag, soc_version)
    export_vae(pipeline, save_dir, batch_size, flag, vae_path, soc_version)
    # controlnet功能，只支持800IA2单卡不带unetcache
    export_unet(pipeline, save_dir, batch_size * 2, flag, soc_version)
    export_control(pipeline, save_dir, controlnet_path, conditioning_scale, flag, soc_version, batch_size)


def main():
    args = parse_arguments()
    mindietorch.set_device(args.device)
    export(args.model, args.controlnet_model, args.vae_model, args.output_dir,
           args.batch_size, args.conditioning_scale,
           args.flag, args.soc)
    print("Done.")


if __name__ == "__main__":
    main()