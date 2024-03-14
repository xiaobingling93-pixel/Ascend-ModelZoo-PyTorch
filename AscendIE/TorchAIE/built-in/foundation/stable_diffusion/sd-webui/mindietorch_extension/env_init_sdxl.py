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
from omegaconf import ListConfig, OmegaConf
from ldm.util import instantiate_from_config
from modules import shared, paths
from config import NpuConfig

class UnetExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.unet_model = model

    def forward(self, x, timesteps, context, y):
        return self.unet_model(x, timesteps, context, y)

def export_unet(sd_config, unet_model, save_dir, batch_size, soc_version):
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)
    
    params = sd_config.model.params.network_config.params
    width_height = 1024

    sample_size = width_height // 8
    in_channels = params.in_channels
    context_dim = params.context_dim
    max_position_embeddings = 77
    adm_in_channels = params.adm_in_channels
    dummy_input = (
        torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
        torch.ones([batch_size], dtype=torch.float32),
        torch.ones(
            [batch_size, max_position_embeddings, context_dim], dtype=torch.float32
        ),
        torch.ones([batch_size, adm_in_channels], dtype=torch.float32),
    )

    traced_model = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    if not os.path.exists(traced_model):
        print("Exporting the SDXL image information creater...")
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
            mindietorch.Input((batch_size,), dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input(
                (batch_size, max_position_embeddings, context_dim),
                dtype=mindietorch.dtype.FLOAT
            ),
            mindietorch.Input((batch_size, adm_in_channels), dtype=mindietorch.dtype.FLOAT)
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

def init_model_xl(device):
    if NpuConfig.Duo:
        soc_version = "Ascend310P3"
    elif NpuConfig.A2:
        soc_version = "Ascend910B4"

    mindietorch.set_device(device)
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(cur_dir_path, "models")
    save_dir_sdxl = os.path.join(save_dir, "models-sdxl")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_sdxl):
        os.makedirs(save_dir_sdxl)

    unet_sdxl = os.path.join(save_dir_sdxl, "unet")
    batch_size = 2
    compiled_unet_model = os.path.join(unet_sdxl, f"unet_bs{batch_size}_compiled.pt")
    if not os.path.exists(compiled_unet_model):
        sd_xl_repo_configs_path = os.path.join(paths.paths['Stable Diffusion XL'], "configs", "inference")
        checkpoint_config = os.path.join(sd_xl_repo_configs_path, "sd_xl_base.yaml")
        sd_config = OmegaConf.load(checkpoint_config)
        unet_model = instantiate_from_config(sd_config.model.params.network_config)
        export_unet(sd_config, unet_model, save_dir_sdxl, batch_size, soc_version)

    mindietorch.finalize()

    return save_dir_sdxl
    