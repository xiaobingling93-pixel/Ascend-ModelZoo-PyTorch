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
from diffusers import StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
import mindietorch
from mindietorch import _enums

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
        "--base_model_path",
        type=str,
        default="./stable-diffusion-v1-5",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="./sd-vae-ft-mse",
        help="vae_model_path.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="./image_encoder",
        help="image_encoder_path.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "--soc",
        choices=["Duo", "A2"],
        default="Duo",
        help="soc version.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device",
    )

    return parser.parse_args()

class ImageEncoderExport(torch.nn.Module):
    def __init__(self, image_encoder_model):
        super().__init__()
        self.image_encoder_model = image_encoder_model

    def forward(self, x):
        return self.image_encoder_model(x)[0]

def export_image_encoder(sd_pipeline, args):
    print("Exporting the image encoder...")
    image_path = os.path.join(args.output_dir, "image_encoder")
    if not os.path.exists(image_path):
        os.makedirs(image_path, mode=0o640)
    batch_size = args.batch_size
    image_encoder_pt_path = os.path.join(image_path, f"image_encoder_bs{batch_size}.pt")
    image_encoder_compiled_path = os.path.join(image_path, f"image_encoder_bs{batch_size}_compiled.ts")

    image_encoder_model = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to('cpu')
    if not os.path.exists(image_encoder_pt_path):
        dummy_input = torch.ones([1, 3, 224, 224], dtype=torch.float32)
        image_export = ImageEncoderExport(image_encoder_model)
        torch.jit.trace(image_export, dummy_input).save(image_encoder_pt_path)
    if not os.path.exists(image_encoder_compiled_path):
        model = torch.jit.load(image_encoder_pt_path).eval()
        compiled_image_model = (
                mindietorch.compile(model,
                                    inputs=[mindietorch.Input((1, 3, 224, 224),
                                                            dtype=mindietorch.dtype.FLOAT)],
                                    allow_tensor_replace_int=True,
                                    require_full_compilation=True,
                                    truncate_long_and_double=True,
                                    precision_policy=_enums.PrecisionPolicy.FP16,
                                    soc_version=soc_version,
                                    optimization_level=0))
        torch.jit.save(compiled_image_model, image_encoder_compiled_path)

class ClipExport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x):
        return self.clip_model(x)[0]
    
def export_clip(sd_pipeline, args):
    print("Exporting the text encoder...")
    clip_path = os.path.join(args.output_dir, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o640)
    batch_size = args.batch_size
    clip_pt_path = os.path.join(clip_path, f"clip_bs{batch_size}.pt")
    clip_compiled_path = os.path.join(clip_path, f"clip_bs{batch_size}_compiled.ts")

    clip_model = sd_pipeline.text_encoder
    max_position_embeddings = clip_model.config.max_position_embeddings

    if not os.path.exists(clip_pt_path):
        dummy_input = torch.ones([batch_size, max_position_embeddings], dtype=torch.int64)
        clip_export = ClipExport(clip_model)
        torch.jit.trace(clip_export, dummy_input).save(clip_pt_path)
    if not os.path.exists(clip_compiled_path):
        model = torch.jit.load(clip_pt_path).eval()
        compiled_clip_model = (
                mindietorch.compile(model,
                                    inputs=[mindietorch.Input((batch_size, max_position_embeddings),
                                                            dtype=mindietorch.dtype.INT64)],
                                    allow_tensor_replace_int=True,
                                    require_full_compilation=True,
                                    truncate_long_and_double=True,
                                    precision_policy=_enums.PrecisionPolicy.FP16,
                                    soc_version=soc_version,
                                    optimization_level=0))
        torch.jit.save(compiled_clip_model, clip_compiled_path)

class UnetExportInit(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet_model(sample, timestep, encoder_hidden_states)[0]
    
def export_unet_init(sd_pipeline, args):
    print("Exporting the image information creater...")
    unet_path = os.path.join(args.output_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    batch_size = args.batch_size * 2
    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compiled.ts")

    unet_model = sd_pipeline.unet
    clip_model = sd_pipeline.text_encoder

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size = clip_model.config.hidden_size
    max_position_embeddings = 81

    if not os.path.exists(unet_pt_path):
        dummy_input = (
            torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.ones([batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32),
        )
        unet = UnetExportInit(unet_model).eval()
        torch.jit.trace(unet, dummy_input).save(unet_pt_path)
    if not os.path.exists(unet_compiled_path):
        model = torch.jit.load(unet_pt_path).eval()
        compiled_unet_model = (
                mindietorch.compile(model,
                                    inputs=[mindietorch.Input((batch_size,
                                                               in_channels,
                                                               sample_size,
                                                               sample_size),
                                                            dtype=mindietorch.dtype.FLOAT),
                                            mindietorch.Input((1,),
                                                            dtype=mindietorch.dtype.INT64),
                                            mindietorch.Input((batch_size,
                                                               max_position_embeddings,
                                                               encoder_hidden_size),
                                                            dtype=mindietorch.dtype.FLOAT)],
                                    allow_tensor_replace_int=True,
                                    require_full_compilation=True,
                                    truncate_long_and_double=True,
                                    precision_policy=_enums.PrecisionPolicy.FP16,
                                    soc_version=soc_version,
                                    optimization_level=0))
        torch.jit.save(compiled_unet_model, unet_compiled_path)

class VaeExport(torch.nn.Module):
    def __init__(self, vae_model, scaling_factor):
        super().__init__()
        self.vae_model = vae_model
        self.scaling_factor = scaling_factor

    def forward(self, latents):
        latents = 1 / self.scaling_factor * latents
        image = self.vae_model.decode(latents)[0]
        image = (image / 2 + 0.5)
        return image.permute(0, 2, 3, 1)
    
def export_vae(sd_pipeline, args):
    print("Exporting the image decoder...")
    vae_path = os.path.join(args.output_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    batch_size = args.batch_size
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    vae_compiled_path = os.path.join(vae_path, f"vae_bs{batch_size}_compiled.ts")

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet
    scaling_factor = vae_model.config.scaling_factor
    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels

    if not os.path.exists(vae_pt_path):
        dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size])
        vae_export = VaeExport(vae_model,scaling_factor)
        torch.jit.trace(vae_export, dummy_input).save(vae_pt_path)
    if not os.path.exists(vae_compiled_path):
        model = torch.jit.load(vae_pt_path).eval()
        compiled_vae_model = (
                mindietorch.compile(model,
                                    inputs=[mindietorch.Input((batch_size,
                                                               in_channels,
                                                               sample_size,
                                                               sample_size),
                                                            dtype=mindietorch.dtype.FLOAT)],
                                    allow_tensor_replace_int=True,
                                    require_full_compilation=True,
                                    truncate_long_and_double=True,
                                    precision_policy=_enums.PrecisionPolicy.FP16,
                                    soc_version=soc_version,
                                    optimization_level=0))
        torch.jit.save(compiled_vae_model, vae_compiled_path)

def export(args):
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to("cpu")
    pipeline = StableDiffusionPipeline.from_pretrained(args.base_model_path, vae=vae).to("cpu")
    mindietorch.set_device(args.device)
    export_image_encoder(pipeline, args)
    export_clip(pipeline, args)
    export_vae(pipeline, args)
    export_unet_init(pipeline, args)
    mindietorch.finalize()

def main(args):
    export(args)
    print("Done!")

if __name__ == "__main__":
    args = parse_arguments()
    if args.soc == "Duo":
        soc_version = "Ascend310P3"
    elif args.soc == "A2":
        soc_version = "Ascend910B4"
    main(args)