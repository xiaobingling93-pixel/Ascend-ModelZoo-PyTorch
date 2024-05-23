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
import time
import os
from background_session import BackgroundInferSession
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from ais_bench.infer.interface import InferSession


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=256, type=int, choices=[256, 512], help='image size')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--num_sampling_steps', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model', required=True, metavar='DIR', help='path to dit model')
    parser.add_argument('--vae', required=True, metavar='DIR', help='path to vae model')
    parser.add_argument('--parallel', action="store_true", help='use parallel during inference')
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument(
        "--results",
        type=str,
        default="./results",
        help="Path of directory to save all class images"
    )

    return parser.parse_args()


def infer(x, t, y, cfg_scale, sessions):
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)

    combined = combined.type(torch.FloatTensor)
    t = t.type(torch.LongTensor)
    y = y.type(torch.LongTensor)

    if args.parallel:
        combined, combined_ = combined.chunk(2)
        t, t_ = t.chunk(2)
        y, y_ = y.chunk(2)
        sessions[1].infer_asyn([combined_, t_, y_])

    model_out = torch.from_numpy(sessions[0].infer([combined, t, y])[0])

    if args.parallel:
        out = torch.from_numpy(sessions[1].wait_and_get_outputs()[0])
        model_out = torch.cat([model_out, out])
 
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)


def encoder_for_dit(class_labels, sessions, vae, diffusion, device):
    torch.manual_seed(args.seed)
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
    model_kwargs["sessions"] = sessions

    start = time.time()
    # Sample images:
    samples = diffusion.p_sample_loop(
        infer, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    # Remove null class samples
    samples, _ = samples.chunk(2, dim=0)
    samples = torch.tensor(vae.infer([(samples / vae.scaling_factor)])[0])
    end = time.time()
    print(f"sample time is: {(end - start):.2f}s")

    # Save and display images:
    save_image(
        samples,
        f"{args.results}/sample_{class_labels[0]:06d}.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )


def main():
    if not os.path.exists(args.results):
        os.makedirs(args.results, mode=0o640)

    # Setup PyTorch:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = InferSession(args.device_id, args.vae)
    vae_path = f"stabilityai/sd-vae-ft-{args.vae.split('.')[-2].split('_')[-1]}"
    vae.scaling_factor = AutoencoderKL.from_pretrained(vae_path).to("cpu").config.scaling_factor

    sessions = []
    session = InferSession(args.device_id, args.model)
    if args.parallel:
        session_bg = BackgroundInferSession.clone(session, args.device_id + 1, [args.model, None])
        sessions = [session, session_bg]
    else:
        sessions = [session]

    # Labels to condition the model with (feel free to change): 
    if args.class_label == -1 :
        for i in range(1000):
            encoder_for_dit([i], sessions, vae, diffusion, device)
    else:
        encoder_for_dit([args.class_label], sessions, vae, diffusion, device)

    if args.parallel:
        session_bg.stop()


if __name__ == "__main__":
    args = parse_arguments()
    main()