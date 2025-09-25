#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from functools import reduce
import torch


class AdaStep:
    """
    The Adastep class is designed to optimize the sampling process in a diffusion model,
    """
    def __init__(self, skip_thr=0.015, max_skip_steps=4, decay_ratio=0.99,
                 device="npu", forward_value=None, step_value=None):
        """
        Args:
        skip_thr (float): The threshold for determining whether to skip a step based on the change in latent variables.
            Recommended values are between 0.01 and 0.015. Default is 0.015.
        max_skip_steps (int): The maximum number of consecutive steps that can be skipped.
            Recommended values are between 3 and 4. Default is 4.
        decay_ratio (float): The decay ratio for the skip threshold, which is used to dynamically adjust
            the threshold over time. Recommended values are between 0.95 and 0.99. Default is 0.99.
        device (str): The device on which the computations will be performed. Default is "npu".
        """

        # recommand 0.01(skip around 35 steps) ~ 0.015(skip around 50 steps)
        self.skip_thr = skip_thr   
        # recommand 3 ~ 4
        self.max_skip_steps = max_skip_steps    
        # recommand 0.95 ~ 0.99
        self.decay_ratio = decay_ratio
        self.device = device
        self.if_skip = self.max_skip_steps > 0
        self.reset_status()

        self.forwardretype = forward_value
        self.stepretype = step_value

    def __call__(self, transformer, *model_args, **model_kwargs):
        """
        Args:
        transformer (Module): The Module that works as the DiT and returns the noise prediction.
        model_args (tuple): The arguments to be passed to the transformer Module forword.
        model_kwargs (dict): The keyword arguments to be passed to the transformer Module forword.
        Returns:
        The noise prediction from the transformer function.
        """
        if self.if_skip and torch.all(self.skip_vote):
            return self._return_output(self.skip_noise_pred, self.forwardretype)
        
        noise_pred = transformer(*model_args, **model_kwargs)
        if not self.forwardretype: 
            if isinstance(noise_pred, tuple):
                self.forwardretype = tuple
            elif isinstance(noise_pred, torch.Tensor):
                self.forwardretype = torch.Tensor
            else:
                raise (ValueError, "Transformer needs return a tuple whose first element is the result, "
                        "or return a tensor. In other cases, please enter `forward_value`.")
        self.skip_noise_pred = self._get_input(noise_pred, self.forwardretype)
        return noise_pred
    
    @staticmethod
    def _get_input(input_value, inp_type):
        if isinstance(inp_type, type):
            if inp_type is tuple:
                return input_value[0]
            else:
                return input_value
        else:
            return input_value[inp_type]
        
    @staticmethod
    def _return_output(output, outptype):
        if isinstance(outptype, type):
            if outptype is tuple:
                return (output,)
            else:
                return output
        elif isinstance(outptype, str):
            return {outptype: output}
        else:
            return (0,) * outptype + (output,)     
    
    def set_param(self, skip_thr=None, max_skip_steps=None, decay_ratio=None, device=None):
        """
        To set the parameters of the AdaStep class.
        """
        self.skip_thr = skip_thr or self.skip_thr
        self.max_skip_steps = max_skip_steps or self.max_skip_steps
        self.decay_ratio = decay_ratio or self.decay_ratio
        if device:
            self.device = device
            self.skip_vote.to(self.device)
        self.if_skip = self.max_skip_steps > 0

    def reset_status(self):
        """
        Reset the status of the AdaStep class.
        """
        self.skip_mask = [False]
        self.skip_latents_diff = []
        self.skip_noise_pred = None
        self.skip_prev_latents = 0
        self.skip_vote = torch.tensor([False], dtype=torch.bool, device=self.device)

    def update_strategy(self, latents, sequence_parallel=False, sp_group=None):
        """
        Update the strategy for skipping steps based on the change in latents.
        """
        if not self.stepretype:
            if isinstance(latents, tuple):
                self.stepretype = tuple
            elif isinstance(latents, torch.Tensor):
                self.stepretype = torch.Tensor
            else:
                raise (ValueError, "step needs return a tuple whose first element is the result, "
                        "or return a tensor. In other cases, please enter `step_value`.")
        if self.if_skip:
            latents = self._get_input(latents, self.stepretype)
            diff = latents - self.skip_prev_latents
            self.skip_latents_diff.append(diff.abs().mean())
            if len(self.skip_latents_diff) >= 3:
                self.skip_mask.append(self._estimate())

            self.skip_prev_latents = latents 

            mask_value = self.skip_mask[-1]
            mask_value = torch.tensor([mask_value], dtype=torch.bool, device=self.device)
            if sequence_parallel:
                skip_vote = torch.zeros(torch.distributed.get_world_size(sp_group), 
                                        dtype=torch.bool, device=self.device)
                torch.distributed.all_gather_into_tensor(skip_vote, mask_value, group=sp_group)
            else:
                skip_vote = mask_value
            self.skip_vote = skip_vote

    def _estimate(self):
        # `self.skip_latents_diff[-1]` refers to the most recent difference in latent variables.
        cur_diff = self.skip_latents_diff[-1]
        # `self.skip_latents_diff[-2]` refers to the second most recent difference in latent variables.
        prev_diff = self.skip_latents_diff[-2]
        # `self.skip_latents_diff[-3]` refers to the third most recent difference in latent variables.
        prev_prev_diff = self.skip_latents_diff[-3]

        self.skip_thr = self.skip_thr * self.decay_ratio

        if len(self.skip_mask) >= self.max_skip_steps and \
            all(self.skip_mask[-self.max_skip_steps:]):
            return False

        if abs((cur_diff + prev_prev_diff) / 2 - prev_diff) <= prev_diff * self.skip_thr:
            return True
        return False
    

class SamplingOptm:
    def __init__(self, pipe, dit_forward="transformer.forward", scheduler_step="scheduler.step", 
                    forward_value=None, step_value=None, parallel=False, group=None, config=None):
        self.parallel = parallel
        self.group = group
        self.skip = False
        if config and config["method"] == "AdaStep":
            self.skip = True
            ditforward_lst = dit_forward.split(".")
            schedulerstep_lst = scheduler_step.split(".")
            self.pipe = pipe

            self.ori_forward = reduce(getattr, ditforward_lst, self.pipe) # getattr(self.pipe, )dit_forward.split(".")
            self.forward = ditforward_lst.pop()
            self.ori_dit = reduce(getattr, ditforward_lst, self.pipe)

            self.ori_step = reduce(getattr, schedulerstep_lst, self.pipe) 
            self.step = schedulerstep_lst.pop()
            self.ori_scheduler = reduce(getattr, schedulerstep_lst, self.pipe)

            shik_thr = config.get("skip_thr", 0.015)
            max_skip_steps = config.get("max_skip_steps", 4)
            decay_ratio = config.get("decay_ratio", 0.99)
            self.skip_strategy = AdaStep(skip_thr=shik_thr, max_skip_steps=max_skip_steps, decay_ratio=decay_ratio)

    def __enter__(self):
        if self.skip:
            self._sub_forward()
            self._sub_step()

    def __exit__(self, t, v, trace):
        if self.skip:
            self._revert_forward()
            self._revert_step()

    def _sub_forward(self):
        @functools.wraps(self.ori_forward)
        def _optm_forward(*args, **kwargs):
            noise_pred = self.skip_strategy(self.ori_forward, *args, **kwargs)
            return noise_pred
        setattr(self.ori_dit, self.forward, _optm_forward)

    def _sub_step(self):
        @functools.wraps(self.ori_step)
        def _optm_step(*args, **kwargs):
            latents = self.ori_step(*args, **kwargs)
            self.skip_strategy.update_strategy(latents, self.parallel, self.group)
            return latents
        setattr(self.ori_scheduler, self.step, _optm_step)
        
    def _revert_forward(self):
        setattr(self.ori_dit, self.forward, self.ori_forward)

    def _revert_step(self):
        setattr(self.ori_scheduler, self.step, self.ori_step)