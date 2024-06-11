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
import math
import argparse
import multiprocessing
from tqdm import tqdm
import torch
import numpy as np
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from PIL import Image
from ais_bench.infer.interface import InferSession


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, choices=["ema", "mse"], default='mse')
    parser.add_argument('--sample_dir', type=str, default='samples_npu')
    parser.add_argument('--per_proc_batch_size', type=int, default=32)
    parser.add_argument('--num_fid_samples', type=int, default=50_000)
    parser.add_argument('--image_size', default=256, type=int, choices=[256, 512], help='image size')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--num_sampling_steps', type=int, default=250)
    parser.add_argument('--global_seed', type=int, default=0)
    parser.add_argument('--model', required=True, metavar='DIR', help='path to model')
    parser.add_argument('--device', type=list, default=[0])

    return parser.parse_args()


def dit_infer_npu(x, t, y, cfg_scale, session):
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


def diffusion_samples(id):
    if DIT_INPUTS.empty():
        om_event.wait()

    session = InferSession(id, args.model)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    while DIT_INPUTS.qsize() > 0:
        input = DIT_INPUTS.get()
        z = input["z"]
        model_kwargs = input["model_kwargs"]
        model_kwargs["session"] = session
        # start loop
        print("start diffusion.p_sample_loop, DIT_INPUTS.qsize() is ", DIT_INPUTS.qsize())
        samples = diffusion.p_sample_loop(
            dit_infer_npu, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        print("diffusion.p_sample_loop end, DIT_INPUTS.qsize() is ", DIT_INPUTS.qsize())
        DIT_OUTPUTS.put(samples)
        out_event.set()


def vae_for_dit(vae):
    if DIT_OUTPUTS.empty():
        out_event.wait()

    index = 0
    while DIT_OUTPUTS.qsize() > 0:
        samples = DIT_OUTPUTS.get()
        # Remove null class samples
        samples, _ = samples.chunk(2, dim=0)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        # samples value range is (-1, 1), (samples // 2 + 0.5) * 255
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(device, dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for sample in samples:
            Image.fromarray(sample).save(f"{args.sample_dir}/{index:06d}.png")
            index += 1


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Build a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)

    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def encoder_for_dit(id):
    n = args.per_proc_batch_size // 2
    pbar = tqdm(range(math.ceil(args.num_fid_samples / n)))

    for _ in pbar:
        # Sample inputs:default model.in_channels is 4, latent_size = args.image_size // 8
        z = torch.randn(n, 4, args.image_size // 8, args.image_size // 8, device=device)
        z = torch.cat([z, z], 0)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        DIT_INPUTS.put(dict(z=z, model_kwargs=model_kwargs))
        om_event.set()


def set_config():
    torch.set_grad_enabled(False)

    #  Create folder to save samples:
    folder_name = f"dit-size-{args.image_size}-vae-{args.vae}-seed-{args.global_seed}"
    args.sample_dir = f"{args.sample_dir}/{folder_name}"
    os.makedirs(args.sample_dir, exist_ok=True)
    print(f"Saving .png samples at {args.sample_dir}")


def create_process_pool():
    processes_num = len(args.device) + 2
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    with multiprocessing.Pool(processes=processes_num) as pool:
        pool.apply_async(encoder_for_dit, [1])
        pool.map_async(diffusion_samples, [args.device[x - 2] for x in range(2, processes_num)])
        pool.apply(vae_for_dit, [vae])
    
    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    create_npz_from_sample_folder(args.sample_dir, args.num_fid_samples)


if __name__ == "__main__":
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DIT_INPUTS = multiprocessing.Queue()
    DIT_OUTPUTS = multiprocessing.Queue()
    om_event = multiprocessing.Event()
    out_event = multiprocessing.Event()

    set_config()
    create_process_pool()