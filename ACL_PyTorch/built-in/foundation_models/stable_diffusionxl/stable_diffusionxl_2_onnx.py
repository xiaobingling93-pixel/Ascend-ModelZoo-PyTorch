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
        help="Path of directory to save ONNX models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
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
        default=7.5, 
        help="guidance_scale"
    )

    return parser.parse_args()


class NewDdim(nn.Module):
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50, alphas_cumprod=None,
                 guidance_scale=7.5, alpha_prod_t_prev_cache=None):
        super(NewDdim, self).__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.alphas_cumprod = alphas_cumprod
        self.guidance_scale = guidance_scale
        self.alpha_prod_t_prev_cache = alpha_prod_t_prev_cache

    def forward(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        step_index: int):
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        model_output = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_prod_t_prev_cache[step_index]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return(prev_sample,)


def export_ddim(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, steps: int, guidance_scale: float) -> None:
    print("Exporting the ddim...")
    ddim_path = os.path.join(save_dir, "ddim")
    if not os.path.exists(ddim_path):
        os.makedirs(ddim_path, mode=0o744)
    
    dummy_input = (
                   torch.randn(2, 4, 128, 128),
                   torch.tensor(981),
                   torch.randn(1, 4, 128, 128),
                   torch.tensor(0)
                   )
    scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)
    scheduler.set_timesteps(steps, device="cpu")

    timesteps = scheduler.timesteps[:steps]
    alpha_prod_t_prev_cache = []
    for timestep in timesteps:
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        alpha_prod_t_prev_cache.append(alpha_prod_t_prev)

    new_ddim = NewDdim(
        num_train_timesteps=scheduler.config.num_train_timesteps,
        num_inference_steps=scheduler.num_inference_steps,
        alphas_cumprod=scheduler.alphas_cumprod,
        guidance_scale=guidance_scale,
        alpha_prod_t_prev_cache=torch.tensor(alpha_prod_t_prev_cache)
    )

    new_ddim.eval()
    torch.onnx.export(
        new_ddim,
        dummy_input,
        os.path.join(ddim_path, "ddim.onnx"),
        input_names=["noises", "timestep", "latents", "step_index"],
        output_names=["out_latents"],
        opset_version=11,
        verbose=False
    )


def export_encoder(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size:int) -> None:
    print("Exporting the text encoder...")
    encoder_path = os.path.join(save_dir, "text_encoder")
    if not os.path.exists(encoder_path):
        os.makedirs(encoder_path, mode=0o744)

    encoder_model = sd_pipeline.text_encoder

    max_position_embeddings = encoder_model.config.max_position_embeddings
    dummy_input = (
        torch.ones([batch_size, max_position_embeddings], dtype=torch.int64),
        None,
        None,
        None,
        True
    )

    torch.onnx.export(
        encoder_model,
        dummy_input,
        os.path.join(encoder_path, "text_encoder.onnx"),
        input_names=["prompt"],
        output_names=["text_embeddings"],
        opset_version=11,
    )

    print("Exporting the text encoder 2...")
    encoder_2_model = sd_pipeline.text_encoder_2

    torch.onnx.export(
        encoder_2_model,
        dummy_input,
        os.path.join(encoder_path, "text_encoder_2.onnx"),
        input_names=["prompt"],
        output_names=["text_embeddings"],
        opset_version=11,
    )


def export_unet(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o744)

    unet_model = sd_pipeline.unet
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings

    dummy_input = (
        torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
        torch.ones([1], dtype=torch.int64),
        torch.ones(
            [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
        ),
        None,
        None,
        None,
        None,
        {
            "text_embeds": torch.ones([batch_size, encoder_hidden_size_2], dtype=torch.float32),
            "time_ids": torch.ones([batch_size, 6], dtype=torch.float32)
        },
        {}
    )

    torch.onnx.export(
        unet_model,
        dummy_input,
        os.path.join(unet_path, f"unet.onnx"),
        input_names=["latent_model_input", "t", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["sample"],
        opset_version=11,
    )


def export_vae(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o744)

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels

    dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size])

    torch.onnx.export(
        vae_model.decoder,
        dummy_input,
        os.path.join(vae_path, "vae.onnx"),
        input_names=["latents"],
        output_names=["image"],
        opset_version=11,
    )


def export_onnx(model_path: str, save_dir: str, batch_size:int, steps:int, guidance_scale:float) -> None:
    pipeline = StableDiffusionXLPipeline.from_pretrained(model_path).to("cpu")

    export_encoder(pipeline, save_dir, batch_size)

    export_unet(pipeline, save_dir, batch_size * 2)

    export_vae(pipeline, save_dir, batch_size)

    export_ddim(pipeline, save_dir, steps, guidance_scale)


def main():
    args = parse_arguments()
    export_onnx(args.model, args.output_dir, args.batch_size, args.steps, args.guidance_scale)
    print("Done.")


if __name__ == "__main__":
    main()