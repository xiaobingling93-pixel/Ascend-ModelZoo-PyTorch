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

import argparse
import os

import torch
from cldm.model import create_model, load_state_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="./base_models/control_sd15_canny.pth",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--control_path",
        type=str,
        default="./models",
        help="path or name of the control.",
    )
    parser.add_argument(
        "--sd_path",
        type=str,
        default="./models",
        help="Path or name of the sd.",
    )

    return parser.parse_args()


class ControlNetExport(torch.nn.Module):
    def __init(self, control_model):
        super().__init__()
        self.control_model = control_model

    def forward(self, x, hint, timesteps, context):
        return self.control_model(x, hint, timesteps, context)


def export_control(model, save_path):
    control_path = os.path.join(save_path, "control")
    if not os.path.exists(control_path):
        os.makedirs(control_path, mode=0o744)
    model = model.control_model.eval()
    dummy_input = (
        torch.ones([1, 4, 64, 72], dtype=torch.float32),
        torch.ones([1, 3, 512, 576], dtype=torch.float32),
        torch.ones([1], dtype=torch.int32),
        torch.ones([1, 77, 768], dtype=torch.float32),
    )
    model_export = ControlNetExport(model).eval()
    torch.jit.trace(model_export, dummy_input).save(
        os.path.join(control_path, "control_pt")
    )


class SDExport(torch.nn.Module):
    def __init(self, sd_model):
        super().__init__()
        self.sd_model = sd_model

    def forward(
        self,
        x,
        timesteps,
        context,
        control_0=None,
        control_1=None,
        control_2=None,
        control_3=None,
        control_4=None,
        control_5=None,
        control_6=None,
        control_7=None,
        control_8=None,
        control_9=None,
        control_10=None,
        control_11=None,
        control_12=None,
    ):
        control = [
            control_0,
            control_1,
            control_2,
            control_3,
            control_4,
            control_5,
            control_6,
            control_7,
            control_8,
            control_9,
            control_10,
            control_11,
            control_12,
        ]
        return self.control_model(x, timesteps, context, control)[0]


def export_sd(model, save_path):
    sd_path = os.path.join(save_path, "sd")
    if not os.path.exists(sd_path):
        os.makedirs(sd_path, mode=0o744)
    model = model.model.diffusion_model.eval()
    dummy_input = (
        torch.ones([1, 4, 64, 72], dtype=torch.float32),
        torch.ones([1], dtype=torch.int32),
        torch.ones([1, 77, 768], dtype=torch.float32),
        torch.ones([1, 320, 64, 72], dtype=torch.float32),
        torch.ones([1, 320, 64, 72], dtype=torch.float32),
        torch.ones([1, 320, 64, 72], dtype=torch.float32),
        torch.ones([1, 320, 32, 36], dtype=torch.float32),
        torch.ones([1, 640, 32, 36], dtype=torch.float32),
        torch.ones([1, 640, 32, 36], dtype=torch.float32),
        torch.ones([1, 640, 16, 18], dtype=torch.float32),
        torch.ones([1, 1280, 16, 18], dtype=torch.float32),
        torch.ones([1, 1280, 16, 18], dtype=torch.float32),
        torch.ones([1, 1280, 8, 9], dtype=torch.float32),
        torch.ones([1, 1280, 8, 9], dtype=torch.float32),
        torch.ones([1, 1280, 8, 9], dtype=torch.float32),
        torch.ones([1, 1280, 8, 9], dtype=torch.float32),
    )
    model_export = SDExport(model).eval()
    torch.jit.trace(model_export, dummy_input).save(os.path.join(sd_path, "sd.pt"))


def main():
    args = parse_arguments()
    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(args.base_model, location="cpu"))
    if not os.path.exists(args.control_path):
        os.makedirs(args.control_path, mode=0o744)
    if not os.path.exists(args.sd_path):
        os.makedirs(args.sd_path, mode=0o744)
    export_control(model, args.control_path)
    print("control model done")
    export_sd(model, args.sd_path)
    print("sd model done")


if __name__ == "__main__":
    main()
