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
import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from ais_bench.infer.interface import InferSession


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, choices=["ema", "mse"], default='mse')
    parser.add_argument('--image_size', default=256, type=int, choices=[256, 512], help='image size')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--cfg_scale', type=float, default=4.0)
    parser.add_argument('--num_sampling_steps', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model', required=True, metavar='DIR', help='path to model')

    return parser.parse_args()


def infer(x, t, y, cfg_scale, session):
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)

    combined = combined.type(torch.FloatTensor)
    t = t.type(torch.LongTensor)
    y = y.type(torch.LongTensor)

    model_out = torch.from_numpy(session.infer([combined, t, y])[0])

    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)


def main():
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to("cpu")

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    latent_size = args.image_size // 8
    # default inchannel is 4
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    model_kwargs["session"] = InferSession(args.device_id, args.model)

    # Sample images:
    samples = diffusion.p_sample_loop(
        infer, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    # Remove null class samples
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / vae.config.scaling_factor).sample

    # Save and display images:
    save_image(samples, "sample_npu.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    args = parse_arguments()
    main()