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
from models import DiT_models


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, metavar='DIR', help='path to model')
    parser.add_argument('--save_dir', default='models/onnx', type=str, help='save dir for onnx model')
    parser.add_argument('--image_size', default=256, type=int, help='image size')
    parser.add_argument('--model_name', default='DiT-XL/2', type=str, help='model name for DiT')
    parser.add_argument('--num_classes', default=1000, type=int, help='classes nums')

    return parser.parse_args()


def main():
    args = parse_arguments()

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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    bs_name = "dynamic"
    save_path = os.path.join(args.save_dir, f"dit_{bs_name}_{args.image_size}.onnx")

    torch.onnx.export(model, inputs, save_path, opset_version=14,
                    input_names=["input"], output_names=["output"],
                    dynamic_axes={"input": {0: "-1"}, "input.5": {0: "-1"},
                    "t": {0: "-1"}, "output": {0: "-1"}})


if __name__ == "__main__":
    main()