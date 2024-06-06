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
from diffusers import ControlNetModel, StableDiffusionXLInpaintPipeline, AutoencoderKL
from compile_model import compile_clip, compile_vae, compile_unet_cache,\
     compile_unet_skip, compile_unet_init, compile_img_encode


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
        "--use_cache",
        action="store_true",
        help="Use cache during inference."
    )

    return parser.parse_args()


class ClipExport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, output_hidden_states=True, return_dict=False):
        return self.clip_model(x, output_hidden_states=output_hidden_states,
                               return_dict=return_dict)


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


def export_clip(sd_pipeline: StableDiffusionXLInpaintPipeline, save_dir: str,
                batch_size: int, flag: int, soc_version: str) -> None:
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

    # trace
    trace_clip(sd_pipeline, batch_size, clip_pt_path, clip2_pt_path)

    # compile
    if flag == 0 or flag == 1:
        if not os.path.exists(clip1_compiled_path):
            model = torch.jit.load(clip_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings),
                                        dtype=mindietorch.dtype.INT64)]
            compile_clip(model, inputs, clip1_compiled_path, soc_version)
        if not os.path.exists(clip2_compiled_path):
            model = torch.jit.load(clip2_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, max_position_embeddings),
                                        dtype=mindietorch.dtype.INT64)]
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
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2

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


def export_unet_init(sd_pipeline: StableDiffusionXLInpaintPipeline,
                     save_dir: str, batch_size: int, flag: int,
                     soc_version: str) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_static.ts")
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
    trace_unet_init(sd_pipeline, batch_size, unet_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(unet_compiled_static_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input((batch_size,
                                         max_position_embeddings,
                                         encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((batch_size,
                                         encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((batch_size, 6),
                                        dtype=mindietorch.dtype.FLOAT)]
            compile_unet_init(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((batch_size, in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT)
                             ]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((batch_size, in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size, 6),
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


def export_unet_skip(sd_pipeline: StableDiffusionXLInpaintPipeline, save_dir: str,
                     batch_size: int, flag: int,
                     soc_version: str) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_1.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_1_static.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_1_haveshape.ts")
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
            torch.ones([batch_size, 1280,
                        math.ceil(sample_size / 2), math.ceil(sample_size / 2)],
                       dtype=torch.float32),
        )
        unet = UnetExport(unet_model)
        unet.eval()
        torch.jit.trace(unet, dummy_input).save(unet_pt_path)
    # compile
    if flag == 0:
        if not os.path.exists(unet_compiled_static_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = [mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input((batch_size,
                                         max_position_embeddings,
                                         encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((batch_size,
                                         encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((batch_size, 6),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input(
                          (batch_size, 1280, math.ceil(sample_size / 2),
                           math.ceil(sample_size / 2)),
                          dtype=mindietorch.dtype.FLOAT)]
            compile_unet_skip(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((batch_size, in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input(
                                 (batch_size, 1280, math.ceil(1024 // 8 / 2),
                                  math.ceil(1024 // 8 / 2)),
                                 dtype=mindietorch.dtype.FLOAT),
                             ]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((batch_size, in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input(
                                 (batch_size, 1280, math.ceil(512 // 8 / 2),
                                  math.ceil(512 // 8 / 2)),
                                 dtype=mindietorch.dtype.FLOAT),
                             ]
            inputs.append(inputs_gear_2)
            compile_unet_skip(model, inputs, unet_compiled_path, soc_version)

    else:
        print("This operation is not supported!")


def export_unet_cache(sd_pipeline: StableDiffusionXLInpaintPipeline, save_dir: str,
                      batch_size: int, flag: int,
                      soc_version: str) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}_0.pt")
    unet_compiled_static_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_0_static.ts")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compile_0.ts")
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
            inputs = [mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64),
                      mindietorch.Input((batch_size,
                                         max_position_embeddings,
                                         encoder_hidden_size),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((batch_size,
                                         encoder_hidden_size_2),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((batch_size, 6),
                                        dtype=mindietorch.dtype.FLOAT),
                      mindietorch.Input((1,),
                                        dtype=mindietorch.dtype.INT64)]
            compile_unet_cache(model, inputs, unet_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(unet_compiled_path):
            model = torch.jit.load(unet_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((batch_size,
                                                in_channels, 1024 // 8, 1024 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size, 6),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             ]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((batch_size,
                                                in_channels, 512 // 8, 512 // 8),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((1,),
                                               dtype=mindietorch.dtype.INT64),
                             mindietorch.Input((batch_size,
                                                max_position_embeddings,
                                                encoder_hidden_size),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size,
                                                encoder_hidden_size_2),
                                               dtype=mindietorch.dtype.FLOAT),
                             mindietorch.Input((batch_size, 6),
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


def export_vae(sd_pipeline: StableDiffusionXLInpaintPipeline, save_dir: str,
               batch_size: int, flag: int,
               soc_version: str) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    vae_compiled_static_path = os.path.join(vae_path, f"vae_bs{batch_size}_compile_static.ts")
    vae_compiled_path = os.path.join(vae_path, f"vae_bs{batch_size}_compile.ts")

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


class ImageEncodeExport(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.tile_sample_min_size = vae.tile_sample_min_size
        self.use_tiling = vae.use_tiling
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.tiled_encode = vae.tiled_encode
        self.use_slicing = vae.use_slicing

    def forward(self, x, return_dict=True):
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        return moments


def export_image_encode(sd_pipeline: StableDiffusionXLInpaintPipeline, save_dir: str,
                        batch_size: int, flag: int,
                        soc_version: str) -> None:
    print("Exporting the image decoder...")

    img_encode_path = os.path.join(save_dir, "image_encode")
    if not os.path.exists(img_encode_path):
        os.makedirs(img_encode_path, mode=0o640)
    img_encode_pt_path = os.path.join(img_encode_path, f"image_encode_bs{batch_size}.pt")
    img_encode_compiled_static_path = os.path.join(img_encode_path,
                                                   f"image_encode_bs{batch_size}_compile_static.ts")
    img_encode_compiled_path = os.path.join(img_encode_path, f"image_encode_bs{batch_size}_compile.ts")

    # trace
    vae_model = sd_pipeline.vae
    if not os.path.exists(img_encode_pt_path):
        dummy_input = torch.ones([1, 3, 1024, 1024], dtype=torch.float32)
        vae_export = ImageEncodeExport(vae_model)
        torch.jit.trace(vae_export, dummy_input).save(img_encode_pt_path)

    # compile
    if flag == 0:
        if not os.path.exists(img_encode_compiled_static_path):
            model = torch.jit.load(img_encode_pt_path).eval()
            # 静态shape
            inputs = [mindietorch.Input((1, 3, 1024, 1024), dtype=mindietorch.dtype.FLOAT)]
            compile_img_encode(model, inputs, img_encode_compiled_static_path, soc_version)
    elif flag == 1:
        if not os.path.exists(img_encode_compiled_path):
            # 动态分档
            model = torch.jit.load(img_encode_pt_path).eval()
            inputs = []
            inputs_gear_1 = [mindietorch.Input((1, 3, 1024, 1024), dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_1)
            inputs_gear_2 = [mindietorch.Input((1, 3, 512, 512), dtype=mindietorch.dtype.FLOAT)]
            inputs.append(inputs_gear_2)

            compile_img_encode(model, inputs, img_encode_compiled_path, soc_version)
    else:
        print("This operation is not supported!")


def export(model_path: str, save_dir: str, batch_size: int, flag: int, soc: str, use_cache: bool) -> None:
    if soc == "A2":
        soc_version = "Ascend910B4"
    else:
        print("unsupport soc_version, please check!")
        return

    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(model_path).to("cpu")

    export_clip(pipeline, save_dir, batch_size, flag, soc_version)
    export_vae(pipeline, save_dir, batch_size, flag, soc_version)
    export_image_encode(pipeline, save_dir, batch_size, flag, soc_version)

    if use_cache:
        # 单卡带unetcache
        export_unet_cache(pipeline, save_dir, batch_size * 2, flag, soc_version)
        export_unet_skip(pipeline, save_dir, batch_size * 2, flag, soc_version)
    else:
        # 单卡不带unetcache
        export_unet_init(pipeline, save_dir, batch_size * 2, flag, soc_version)


def main():
    args = parse_arguments()
    export(args.model, args.output_dir, args.batch_size, args.flag, args.soc, args.use_cache)
    print("Done.")


if __name__ == "__main__":
    main()
