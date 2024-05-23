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
import torch
from argparse import Namespace
from models import DiT_models
from diffusers.models import AutoencoderKL


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model
    
    def forward(self, latents):
        return self.vae_model.decode(latents)[0]


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, metavar='DIR', help='path to model')
    parser.add_argument('--save_dir', default='models', type=str, help='save dir for onnx model')
    parser.add_argument('--image_size', default=256, type=int, help='image size')
    parser.add_argument('--model_name', default='DiT-XL/2', type=str, help='model name for DiT')
    parser.add_argument('--num_classes', default=1000, type=int, help='classes nums')
    parser.add_argument('--vae', default='mse', type=str, choices=['ema', 'mse'])

    return parser.parse_args()


def export_dit_onnx(args: Namespace) -> None:
    latent_size = args.image_size // 8
    model = DiT_models[args.model_name](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to("cpu")

    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval()

    bs = 2
    x_input = torch.randn(bs, 4, latent_size, latent_size)
    y_input = torch.randint(low=0, high=args.num_classes, size=(bs,))
    t_input = torch.randint(low=0, high=args.num_classes, size=(bs,))
    inputs = (x_input, y_input, t_input)

    save_path = os.path.join(args.save_dir, "dit_onnx")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"dit_dynamic_{args.image_size}.onnx")

    torch.onnx.export(
        model,
        inputs,
        save_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "-1"}, "input.5": {0: "-1"},
                    "t": {0: "-1"}, "output": {0: "-1"}}
        )


def export_vae_onnx(args: Namespace) -> None:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to("cpu")
    bs = 1
    latent_size = args.image_size // 8
    latents = torch.randn(bs, 4, latent_size, latent_size)

    model = VaeExport(vae)
    model.eval()

    save_path = os.path.join(args.save_dir, "vae_onnx")
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o640)
    save_path = os.path.join(save_path, f"vae_dynamic_{args.image_size}_{args.vae}.onnx")

    torch.onnx.export(
        model,
        latents,
        save_path,
        opset_version=14,
        input_names=["latents"],
        output_names=["image"],
        dynamic_axes={"latents": {0: "-1"}, "image": {0: "-1"}}
        )


def main():
    args = parse_arguments()
    export_dit_onnx(args)
    export_vae_onnx(args)


if __name__ == "__main__":
    main()