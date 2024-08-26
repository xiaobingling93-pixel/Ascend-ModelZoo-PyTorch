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
from dataclasses import dataclass
from typing import List
import mindietorch
from mindietorch import _enums


@dataclass
class CompileParam:
    inputs: List[mindietorch.Input] = None
    soc_version: str = ""
    allow_tensor_replace_int: bool = True
    require_full_compilation: bool = True
    truncate_long_and_double: bool = True
    min_block_size: int = 1


def common_compile(model, compiled_path, compile_param):
    compiled_model = (
        mindietorch.compile(model,
                            inputs=compile_param.inputs,
                            allow_tensor_replace_int=compile_param.allow_tensor_replace_int,
                            require_full_compilation=compile_param.require_full_compilation,
                            truncate_long_and_double=compile_param.truncate_long_and_double,
                            min_block_size=compile_param.min_block_size,
                            soc_version=compile_param.soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0))
    torch.jit.save(compiled_model, compiled_path)


class ClipExport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, output_hidden_states=True, return_dict=False):
        return self.clip_model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)


def compile_clip(model, inputs, clip_compiled_path, soc_version):
    clip_param = CompileParam(inputs, soc_version, True, False, False)
    common_compile(model, clip_compiled_path, clip_param)


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model, scaling_factor, shift_factor):
        super().__init__()
        self.vae_model = vae_model
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

    def forward(self, latents):
        latents = (latents / self.scaling_factor) + self.shift_factor
        image = self.vae_model.decode(latents, return_dict=False)[0]
        return image

def compile_vae(model, inputs, vae_compiled_path, soc_version):
    vae_param = CompileParam(inputs, soc_version)
    common_compile(model, vae_compiled_path, vae_param)


class DiTExport(torch.nn.Module):
    def __init__(self, dit_model):
        super().__init__()
        self.dit_model = dit_model

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
    ):
        return self.dit_model(hidden_states, encoder_hidden_states, pooled_projections,
                              timestep, None, False)[0]


def compile_dit(model, inputs, dit_compiled_path, soc_version):
    dit_param = CompileParam(inputs, soc_version)
    common_compile(model, dit_compiled_path, dit_param)
