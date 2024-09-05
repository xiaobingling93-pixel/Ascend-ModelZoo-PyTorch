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

# Scheduler coefficient, compute coefficient manually in advance to compile scheduler npu model. For details, see:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
SCHEDULER_SIGMAS = torch.tensor([1.0000, 0.9874, 0.9741, 0.9601, 0.9454, 0.9298, 0.9133, 0.8959, 0.8774,
                                 0.8577, 0.8367, 0.8143, 0.7904, 0.7647, 0.7371, 0.7073, 0.6751, 0.6402,
                                 0.6022, 0.5606, 0.5151, 0.4649, 0.4093, 0.3474, 0.2780, 0.1998, 0.1109,
                                 0.0089, 0.0000], dtype=torch.float32)


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


class Scheduler(torch.nn.Module):
    def __init__(self):
        super(Scheduler, self).__init__()
        self.sigmas = SCHEDULER_SIGMAS

    def forward(
            self,
            model_output: torch.FloatTensor,
            sample: torch.FloatTensor,
            step_index: torch.LongTensor
    ):
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        sigma = self.sigmas[step_index]

        sigma_hat = sigma
        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[step_index + 1] - sigma_hat
        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        return prev_sample


def compile_scheduler(model, inputs, scheduler_compiled_path, soc_version):
    scheduler_param = CompileParam(inputs, soc_version, True, True, False)
    common_compile(model, scheduler_compiled_path, scheduler_param)


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


class DiTExportCache(torch.nn.Module):
    def __init__(self, dit_cache_model):
        super().__init__()
        self.dit_cache_model = dit_cache_model

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            cache_param,
            if_skip: int = 0,
            delta_cache: torch.FloatTensor = None,
            delta_cache_hidden: torch.FloatTensor = None,
            use_cache: bool = True,
    ):
        return self.dit_cache_model(hidden_states, encoder_hidden_states, pooled_projections, timestep,
                              cache_param, if_skip, delta_cache, delta_cache_hidden, use_cache,
                              joint_attention_kwargs=None, return_dict=False)


def compile_dit_cache(model, inputs, dit_cache_compiled_path, soc_version):
    dit_cache_param = CompileParam(inputs, soc_version, True, False, True)
    common_compile(model, dit_cache_compiled_path, dit_cache_param)
