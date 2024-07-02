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

import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from download import find_model
import argparse
from argparse import Namespace
import mindietorch
import time
from models_npu import DiT_models

def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./DiT-XL-2-256x256.pt",
        help="Path or name of the pre-trained model."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--parallel", action="store_true", help="Use parallel during inference")
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save models"
    )
    return parser.parse_args()

def warm_up(args, dit_compiled_model, vae_compiled_model):
    batch = 1 if args.parallel else 2
    latent_size = args.image_size // 8
    x1 = torch.ones([batch, 4, latent_size, latent_size], dtype=torch.float32)
    x2 = torch.ones([batch,], dtype=torch.int64)
    x3 = torch.ones([batch,], dtype=torch.int64)
    x4 = torch.ones([1, 4, latent_size, latent_size], dtype=torch.int64)
    count = 5
    stream = mindietorch.npu.Stream(f"npu:{args.device}")
    for _ in range(count):
        with mindietorch.npu.stream(stream):
            dit_out_npu = dit_compiled_model(x1.to(f"npu:{args.device}"),
                                             x2.to(f"npu:{args.device}"),
                                             x3.to(f"npu:{args.device}"))
            stream.synchronize()
        dit_out_cpu = dit_out_npu.to("cpu")

        with mindietorch.npu.stream(stream):
            vae_out_npu = vae_compiled_model(x4.to(f"npu:{args.device}"))
            stream.synchronize()
        vae_out_cpu = vae_out_npu.to("cpu")

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cpu"
    device_id = args.device

    if args.parallel:
        mindie_model_path = f"{args.output_dir}/dit/dit_model_{args.image_size}_parallel_compiled.pt"
    else:
        mindie_model_path = f"{args.output_dir}/dit/dit_model_{args.image_size}_compiled.pt"
    vae_compiled_model_path = f"{args.output_dir}/vae/vae_{args.vae}_{args.image_size}_compiled.pt"
    vae_compiled_model = torch.jit.load(vae_compiled_model_path).eval()
    dit_compiled_model = torch.jit.load(mindie_model_path).eval()

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    class_labels = [args.class_label]

    # Create sampling noise
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    mindietorch.set_device(device_id)
    warm_up(args, dit_compiled_model, vae_compiled_model)
    model.set_npu_model_stream(args.parallel, device_id, args.image_size, mindie_model_path, dit_compiled_model)
    start = time.time()
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )
    samples, _ = samples.chunk(2, dim=0) # Remove null class samples
    samples = vae_compiled_model((samples / 0.18215).to(f"npu:{device_id}")).to('cpu') # 0.18215 is scale factor
    end = time.time()
    print(f"sample time is: {(end-start):.2f}s")
    if args.parallel:
        model.end_asyn()

    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    mindietorch.finalize()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

