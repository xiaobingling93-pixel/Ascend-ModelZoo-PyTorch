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
import mindietorch
from mindietorch import _enums

class ClipExport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x, output_hidden_states=True, return_dict=False):
        return self.clip_model(x, output_hidden_states=output_hidden_states, return_dict=return_dict)

def compile_clip(model, inputs, clip_compiled_path, soc_version):
    compiled_clip_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=False,
                            truncate_long_and_double=False,
                            min_block_size=1,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0))
    torch.jit.save(compiled_clip_model, clip_compiled_path)

class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        return self.vae_model.decoder(latents)

def compile_vae(model, inputs, vae_compiled_path, soc_version):
    compiled_vae_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=True,
                            truncate_long_and_double=True,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0))
    torch.jit.save(compiled_vae_model, vae_compiled_path)

class NewScheduler(torch.nn.Module):
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50, alphas_cumprod=None,
                 guidance_scale=5.0, alpha_prod_t_prev_cache=None):
        super(NewScheduler, self).__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.alphas_cumprod = alphas_cumprod
        self.guidance_scale = guidance_scale
        self.alpha_prod_t_prev_cache = alpha_prod_t_prev_cache

    def forward(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, step_index: int):
        divide_batch = (model_output.shape[0]) // 2
        noise_pred_uncond = model_output[:divide_batch, ..., ..., ...]
        noise_pred_text = model_output[divide_batch:, ..., ..., ...]
        model_output = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_prod_t_prev_cache[step_index]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample

class Scheduler(torch.nn.Module):
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50, alphas_cumprod=None,
                 guidance_scale=5.0, alpha_prod_t_prev_cache=None):
        super(Scheduler, self).__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.alphas_cumprod = alphas_cumprod
        self.guidance_scale = guidance_scale
        self.alpha_prod_t_prev_cache = alpha_prod_t_prev_cache

    def forward(self, noise_pred_uncond: torch.FloatTensor, noise_pred_text: torch.FloatTensor, timestep: int,
                sample: torch.FloatTensor, step_index: int):
        model_output = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_prod_t_prev_cache[step_index]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample

def compile_ddim(model, inputs, scheduler_compiled_path, soc_version):
    compiled_scheduler_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=True,
                            truncate_long_and_double=False,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0))
    torch.jit.save(compiled_scheduler_model, scheduler_compiled_path)

class UnetExport(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(
            self,
            sample,
            timestep,
            encoder_hidden_states,
            text_embeds,
            time_ids,
            if_skip,
            inputCache=None
    ):
        if if_skip:
            return self.unet_model(sample, timestep, encoder_hidden_states,
                                   added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                                   if_skip=if_skip, inputCache=inputCache)[0]
        else:
            return self.unet_model(sample, timestep, encoder_hidden_states,
                                   added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                                   if_skip=if_skip)

def compile_unet_cache(model, inputs, unet_compiled_path, soc_version):
    compiled_unet_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=False,
                            truncate_long_and_double=False,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0))
    torch.jit.save(compiled_unet_model, unet_compiled_path)

def compile_unet_skip(model, inputs, unet_compiled_path, soc_version):
    compiled_unet_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=True,
                            truncate_long_and_double=True,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0))
    torch.jit.save(compiled_unet_model, unet_compiled_path)

class UnetExportInit(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(
            self,
            sample,
            timestep,
            encoder_hidden_states,
            text_embeds,
            time_ids
    ):
        return self.unet_model(sample, timestep, encoder_hidden_states,
                               added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids})[0]

def compile_unet_init(model, inputs, unet_compiled_path, soc_version):
    compiled_unet_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=True,
                            truncate_long_and_double=True,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0))
    torch.jit.save(compiled_unet_model, unet_compiled_path)