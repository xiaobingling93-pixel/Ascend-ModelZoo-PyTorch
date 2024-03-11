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
        default=7.5,
        help="guidance_scale"
    )

    return parser.parse_args()


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

    encoder_model = sd_pipeline.text_encoder
    encoder_2_model = sd_pipeline.text_encoder_2

    max_position_embeddings = encoder_model.config.max_position_embeddings
    print(f'max_position_embeddings: {max_position_embeddings}')

    dummy_input = torch.ones([batch_size, max_position_embeddings], dtype=torch.int64)

    clip_export = ClipExport(encoder_model)
    torch.jit.trace(clip_export, dummy_input).save(os.path.join(clip_path, "clip.pt"))

    clip_export = ClipExport(encoder_2_model)
    torch.jit.trace(clip_export, dummy_input).save(os.path.join(clip_path, "clip_2.pt"))


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
            time_ids
    ):
        return self.unet_model(sample, timestep, encoder_hidden_states,
                               added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids})[0]


def export_unet(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

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
        torch.ones([batch_size, encoder_hidden_size_2], dtype=torch.float32),
        torch.ones([batch_size, 6], dtype=torch.float32)
    )

    unet = UnetExport(unet_model)
    unet.eval()

    torch.jit.trace(unet, dummy_input).save(os.path.join(unet_path, f"unet_bs{batch_size}.pt"))


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        return self.vae_model.decode(latents)[0]


def export_vae(sd_pipeline: StableDiffusionXLPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels

    dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32)
    vae_export = VaeExport(vae_model)
    torch.jit.trace(vae_export, dummy_input).save(os.path.join(vae_path, "vae.pt"))


def export(model_path: str, save_dir: str, batch_size: int) -> None:
    pipeline = StableDiffusionXLPipeline.from_pretrained(model_path).to("cpu")

    export_clip(pipeline, save_dir, batch_size)
    export_vae(pipeline, save_dir, batch_size)
    export_unet(pipeline, save_dir, batch_size * 2)


def main():
    args = parse_arguments()
    export(args.model, args.output_dir, args.batch_size)
    print("Done.")


if __name__ == "__main__":
    main()