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
import sys
import time
import math
import numpy as np
import torch
import mindietorch
from mindietorch import _enums
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from sgm.modules.diffusionmodules.openaimodel import UNetModel as UNetModel_XL
from modules import shared
from diffusers import StableDiffusionPipeline
from config import NpuConfig
from env_init_sd import init_model
from env_init_sdxl import init_model_xl

def replace_torch_aie():

    device_0, device_1 = 0, None
    mindietorch.set_device(device_0)

    model_base_1_5, model_base_2_1, save_dir_1_5, save_dir_2_1 = init_model(device_0)
    save_dir_sdxl = init_model_xl(device_0)
    print("You can generate image now!")

    def mindietorch_unet(self, x, timesteps=None, context=None, y=None, **kwargs):
        if x.shape[-1] != 64:
            return 64
        checkpoint = shared.opts.data['sd_model_checkpoint']
        if "v1-5-pruned-emaonly" in checkpoint:
            unet_model = NpuConfig.compiled_unet_model_1_5
            model_base = model_base_1_5
            unet_path = os.path.join(save_dir_1_5, "unet")
        elif "v2-1_512-ema-pruned" in checkpoint:
            unet_model = NpuConfig.compiled_unet_model_2_1
            model_base = model_base_2_1
            unet_path = os.path.join(save_dir_2_1, "unet")

        if not unet_model:
            batch_size = 2
            unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compiled.pt")

            pipe = StableDiffusionPipeline.from_pretrained(model_base).to('cpu')
            sample_size = pipe.unet.config.sample_size
            in_channels = pipe.unet.config.in_channels
            encoder_hidden_size = pipe.text_encoder.config.hidden_size
            max_position_embeddings = pipe.text_encoder.config.max_position_embeddings

            if "v1-5-pruned-emaonly" in checkpoint:
                NpuConfig.compiled_unet_model_1_5 = torch.jit.load(unet_compiled_path).eval()
            elif "v2-1_512-ema-pruned" in checkpoint:
                NpuConfig.compiled_unet_model_2_1 = torch.jit.load(unet_compiled_path).eval()

        if "v1-5-pruned-emaonly" in checkpoint:
            noise_pred = NpuConfig.compiled_unet_model_1_5(
                x.to(f"npu:{device_0}"),
                timesteps[0][None].type(torch.int64).to(f"npu:{device_0}"),
                context.to(f"npu:{device_0}")
            ).to("cpu")
        elif "v2-1_512-ema-pruned" in checkpoint:
            noise_pred = NpuConfig.compiled_unet_model_2_1(
                x.to(f"npu:{device_0}"),
                timesteps[0][None].type(torch.int64).to(f"npu:{device_0}"),
                context.to(f"npu:{device_0}")
            ).to("cpu")
        
        return noise_pred
    UNetModel.forward = mindietorch_unet

    def mindietorch_unet_xl(self, x, timesteps=None, context=None, y=None, **kwargs):
        checkpoint = shared.opts.data['sd_model_checkpoint']
        if x.shape[-1] != 128:
            print("The width and height should be 1024!")
            return x
        assert "sd_xl_base_1.0" in checkpoint, "Please select correct weight: sd_xl_base_1.0.safetensors"

        unet_model = NpuConfig.compiled_unet_model_xl
        unet_path = os.path.join(save_dir_sdxl, "unet")
        batch_size = 2
        unet_compiled_path = os.path.join(unet_path, f"unet_bs{batch_size}_compiled.pt")

        if not unet_model:
            NpuConfig.compiled_unet_model_xl = torch.jit.load(unet_compiled_path).eval()
        noise_pred = NpuConfig.compiled_unet_model_xl(
            x.to(f"npu:{device_0}"),
            timesteps.to(f"npu:{device_0}"),
            context.to(f"npu:{device_0}"),
            y.to(f"npu:{device_0}")
        ).to("cpu")
        return noise_pred
    UNetModel_XL.forward = mindietorch_unet_xl

        