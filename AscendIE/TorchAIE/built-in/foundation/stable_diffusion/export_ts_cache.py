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

import argparse
import os
from argparse import Namespace

import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler


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
        default="./stable-diffusion-2-1-base",
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


class Clipexport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x):
        return self.clip_model(x)[0]


def export_clip(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the text encoder...")
    clip_path = os.path.join(save_dir, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o640)

    clip_model = sd_pipeline.text_encoder

    max_position_embeddings = clip_model.config.max_position_embeddings
    print(f'max_position_embeddings: {max_position_embeddings}')
    dummy_input = torch.ones([batch_size, max_position_embeddings], dtype=torch.int64)
    clip_export = Clipexport(clip_model)

    torch.jit.trace(clip_export, dummy_input).save(os.path.join(clip_path, "clip.pt"))


class Unetexport(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(self, sample, timestep, encoder_hidden_states, if_skip, inputCache=None):
        if if_skip:
            print("[Unetexport][forward] skip --------")
            return self.unet_model(sample, timestep, encoder_hidden_states, if_skip=if_skip, inputCache=inputCache)[0]
        else:
            print("[Unetexport][forward] cache --------")
            return self.unet_model(sample, timestep, encoder_hidden_states, if_skip=if_skip)


def export_unet(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int, if_skip: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    unet_model = sd_pipeline.unet
    clip_model = sd_pipeline.text_encoder

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size = clip_model.config.hidden_size
    max_position_embeddings = clip_model.config.max_position_embeddings
    
    if if_skip:
        dummy_input = (
            torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.ones(
                [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
            ),
            torch.ones([1], dtype=torch.int64),
            torch.ones([batch_size, 640, sample_size, sample_size], dtype=torch.float32),
        )
    else:
        dummy_input = (
            torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.ones(
                [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
            ),
            torch.zeros([1], dtype=torch.int64),
        )


    unet = Unetexport(unet_model)
    unet.eval()

    torch.jit.trace(unet, dummy_input).save(os.path.join(unet_path, f"unet_bs{batch_size}_{if_skip}.pt"))


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model,scaling_factor):
        super().__init__()
        self.vae_model = vae_model
        self.scaling_factor = scaling_factor

    def forward(self, latents):
        latents = 1 / self.scaling_factor * latents
        image = self.vae_model.decode(latents)[0]
        image = (image / 2 + 0.5)
        return image.permute(0, 2, 3, 1)


def export_vae(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet

    scaling_factor = vae_model.config.scaling_factor
    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels
    dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size])
    vae_export = VaeExport(vae_model,scaling_factor)
    torch.jit.trace(vae_export, dummy_input).save(os.path.join(vae_path, "vae.pt"))


class CatExport(torch.nn.Module):
    def __init__(self, scale_model_input):
        super(CatExport, self).__init__()
        self.scale_model_input = scale_model_input

    def forward(self, latents:torch.FloatTensor, t:torch.FloatTensor):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scale_model_input(latent_model_input, t)
        return latent_model_input


def export_cat(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int) -> None:

    cat_path = os.path.join(save_dir, "cat")
    if not os.path.exists(cat_path):
        os.makedirs(cat_path, mode=0o640)

    ddim_model = DDIMScheduler.from_config(sd_pipeline.scheduler.config)

    dummy_input = (
            torch.ones([batch_size, 4, 64, 64], dtype=torch.float32),
            torch.ones([1], dtype=torch.float32))

    cat_export = CatExport(scale_model_input=ddim_model.scale_model_input)
    cat_export.eval()
    torch.jit.trace(cat_export, dummy_input).save(os.path.join(cat_path, "cat.pt"))


class Scheduler(torch.nn.Module):
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50, alphas_cumprod=None,
                 guidance_scale=7.5, alpha_prod_t_prev_cache=None):
        super(Scheduler, self).__init__()
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

def export_ddim(sd_pipeline: StableDiffusionPipeline, save_dir: str, steps: int, guidance_scale: float) -> None:
    print("Exporting the ddim...")
    ddim_path = os.path.join(save_dir, "ddim")
    if not os.path.exists(ddim_path):
        os.makedirs(ddim_path, mode=0o640)

    ddim_model = sd_pipeline.scheduler
    dummy_input = (
            torch.randn([2, 4, 64, 64], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            torch.randn([1, 4, 64, 64], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
                   )
  
    scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)
    scheduler.set_timesteps(steps, device="cpu")
    
    timesteps = scheduler.timesteps[:steps]
    alpha_prod_t_prev_cache = []
    for timestep in timesteps:
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        alpha_prod_t_prev_cache.append(alpha_prod_t_prev)

    new_ddim = Scheduler(
        num_train_timesteps=scheduler.config.num_train_timesteps,
        num_inference_steps=scheduler.num_inference_steps,
        alphas_cumprod=scheduler.alphas_cumprod,
        guidance_scale=guidance_scale,
        alpha_prod_t_prev_cache=torch.tensor(alpha_prod_t_prev_cache)
    )

    new_ddim.eval()

    torch.jit.trace(new_ddim, dummy_input).save(os.path.join(ddim_path, "ddim.pt"))


def export_onnx(model_path: str, save_dir: str, batch_size:int, steps:int, guidance_scale:float) -> None:
    pipeline = StableDiffusionPipeline.from_pretrained(model_path).to("cpu")

    export_clip(pipeline, save_dir, batch_size)

    # 单卡, unet_cache
    export_unet(pipeline, save_dir, batch_size * 2, 0)
    # 单卡, unet_skip
    export_unet(pipeline, save_dir, batch_size * 2, 1)

    export_vae(pipeline, save_dir, batch_size)

    export_ddim(pipeline, save_dir, steps, guidance_scale)

    export_cat(pipeline, save_dir, batch_size)


def main():
    args = parse_arguments()
    export_onnx(args.model, args.output_dir, args.batch_size, args.steps, args.guidance_scale)
    print("Done.")


if __name__ == "__main__":
    main()