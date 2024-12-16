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

import logging
from typing import List

import torch
from torch.distributions import LogisticNormal

from mindiesd.schedulers.scheduler_utils import DiffusionScheduler

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# some code are inspired by
# https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py

UNIFORM_CONSTANT = "uniform"
LOGIT_NORMAL_CONSTANT = "logit-normal"


class RFlowScheduler(DiffusionScheduler):
    def __init__(
            self,
            num_timesteps: int = 1000,
            num_sampling_steps: int = 30,
    ):

        super().__init__()

        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = False
        self.sample_method = UNIFORM_CONSTANT
        self.loc = 0.0
        self.scale = 1.0
        self.use_timestep_transform = True
        self.transform_scale = 1.0

        # sample method
        if self.sample_method not in [UNIFORM_CONSTANT, LOGIT_NORMAL_CONSTANT]:
            logger.error("sample_method must be either 'uniform' or 'logit-normal'")
            raise ValueError("sample_method must be either 'uniform' or 'logit-normal'")

        if self.sample_method != UNIFORM_CONSTANT and self.use_discrete_timesteps:
            logger.error("Only uniform sampling is supported for discrete timesteps")
            raise ValueError("Only uniform sampling is supported for discrete timesteps")

        if self.sample_method == LOGIT_NORMAL_CONSTANT:
            self.distribution = LogisticNormal(torch.tensor([self.loc]), torch.tensor([self.scale]))
            self.sample_t = self._sample_t_function

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])
        res = timepoints * original_samples + (1 - timepoints) * noise

        return res

    def step(self, pred: torch.FloatTensor,
        timesteps: List[float],
        i: int,
        noise: torch.FloatTensor,
        guidance_scale: float = 7.0):

        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        # update z
        dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
        dt = dt / self.num_timesteps
        noise = noise + v_pred * dt[:, None, None, None, None]
        return noise

    def _sample_t_function(self, x):
        return self.distribution.sample((x.shape[0],))[:, 0].to(x.device)


