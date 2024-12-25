#!/usr/bin/env python
# coding=utf-8
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
import numpy as np

from mindiesd import DiffusionScheduler
from ..utils import randn_tensor


class DDPMScheduler(DiffusionScheduler):

    def __init__(
        self,
        steps_offset: int = 0,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()

        self.steps_offset = steps_offset
        self.num_train_timesteps = num_train_timesteps

        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())


    def set_timesteps(self, num_inference_steps: int = 100, device=None):

        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps += self.steps_offset

        self.timesteps = torch.from_numpy(timesteps).to(device)


    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, generator=None):

        prev_t = self._previous_timestep(timestep)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        # 2. compute predicted original sample from predicted noise also called
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        # 4. Compute predicted previous sample µ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        # 5. Add noise
        variance = 0
        if timestep > 0:
            device = model_output.device
            variance_noise = randn_tensor(model_output.shape,
                                          generator=generator,
                                          device=device,
                                          dtype=model_output.dtype)
            variance = (self._get_variance(timestep) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample


    def _previous_timestep(self, timestep):
        num_inference_steps = (self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps)
        prev_t = timestep - self.num_train_timesteps // num_inference_steps
        return prev_t


    def _get_variance(self, timestep):
        prev_t = self._previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance