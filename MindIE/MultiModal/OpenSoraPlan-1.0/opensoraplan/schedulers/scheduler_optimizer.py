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

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler


def get_scheduler(sample_method):
    if sample_method == 'DDIM':
        scheduler = DDIMScheduler()
    elif sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif sample_method == 'DDPM':
        scheduler = DDPMScheduler()
    elif sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif sample_method == 'HeunDiscrete':
        scheduler = HeunDiscreteScheduler()
    elif sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif sample_method == 'KDPM2AncestralDiscrete':
        scheduler = KDPM2AncestralDiscreteScheduler()
    else:
        raise ValueError('ERROR: wrong sample_method given !!!')
    return scheduler


