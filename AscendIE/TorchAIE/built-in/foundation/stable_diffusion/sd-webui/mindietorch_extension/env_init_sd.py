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
import numpy as np
import torch
import mindietorch
from mindietorch import _enums
from diffusers import StableDiffusionPipeline
from config import NpuConfig

class UnetExport(torch.nn.Module):
    def __init__(self, model):
        super(UnetExport, self).__init__()
        self.unet_model = model

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet_model(sample, timestep, encoder_hidden_states)[0]

def export_unet(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int, soc_version: str) -> None:
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    unet_model = sd_pipeline.unet
    clip_model = sd_pipeline.text_encoder

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size = clip_model.config.hidden_size
    max_position_embeddings = clip_model.config.max_position_embeddings
    dummy_input = (
        torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
        torch.ones([1], dtype=torch.int64),
        torch.ones(
            [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
        ),
    )

    traced_model = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    if not os.path.exists(traced_model):
        print("Exporting the image information creater...")
        unet = UnetExport(unet_model)
        unet.eval()
        model = torch.jit.trace(unet, dummy_input)
        torch.jit.save(model, traced_model)
    else:
        model = torch.jit.load(traced_model).eval()

    compiled_model = os.path.join(unet_path, f"unet_bs{batch_size}_compiled.pt")
    if not os.path.exists(compiled_model):
        print("start compile unet model...")
        unet_input_info = [
            mindietorch.Input(
                (batch_size, in_channels, sample_size, sample_size),
                dtype=mindietorch.dtype.FLOAT
            ),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input(
                (batch_size, max_position_embeddings, encoder_hidden_size),
                dtype=mindietorch.dtype.FLOAT
            )
        ]
        compiled_unet_model = mindietorch.compile(
            model,
            inputs=unet_input_info,
            allow_tensor_replace_int=True,
            require_full_compilation=True,
            truncate_long_and_double=True,
            soc_version=soc_version,
            precision_policy=_enums.PrecisionPolicy.FP16,
            optimization_level=0
        )
        torch.jit.save(compiled_unet_model, compiled_model)

def init_model(device):
    mindietorch.set_device(device)
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(cur_dir_path, "models")
    save_dir_1_5 = os.path.join(save_dir, "models-1-5")
    save_dir_2_1 = os.path.join(save_dir, "models-2-1")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_1_5):
        os.makedirs(save_dir_1_5)
    if not os.path.exists(save_dir_2_1):
        os.makedirs(save_dir_2_1)

    unet_path_1_5 = os.path.join(save_dir_1_5, "unet")
    unet_path_2_1 = os.path.join(save_dir_2_1, "unet")

    weights_1_5 = os.path.join(save_dir, "stable-diffusion-v1-5")
    weights_2_1 = os.path.join(save_dir, "stable-diffusion-2-1-base")

    batch_size = 1
    pipe_1_5 = StableDiffusionPipeline.from_pretrained(weights_1_5).to('cpu')
    pipe_2_1 = StableDiffusionPipeline.from_pretrained(weights_2_1).to('cpu')

    if NpuConfig.Duo:
        soc_version = "Ascend310P3"
    elif NpuConfig.A2:
        soc_version = "Ascend910B4"

    export_unet(pipe_1_5, save_dir_1_5, batch_size * 2, soc_version)
    export_unet(pipe_2_1, save_dir_2_1, batch_size * 2, soc_version)

    mindietorch.finalize()

    return weights_1_5, weights_2_1, save_dir_1_5, save_dir_2_1
    