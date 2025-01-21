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

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch_npu
from einops import rearrange
from tqdm import tqdm


from ..models.parallel_mgr import get_sequence_parallel_state, get_sequence_parallel_size, get_sequence_parallel_rank
from .pipeline_utils import DiffusionPipeline, rescale_noise_cfg, retrieve_timesteps


class OpenSoraPlanPipeline13(DiffusionPipeline):

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        scheduler,
        text_encoder_2=None,
        tokenizer_2=None
    ):
        super().__init__()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder_2 = text_encoder_2
        self._guidance_scale = None

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_samples_per_prompt: Optional[int] = 1,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        guidance_rescale: float = 0.0,
        max_sequence_length: int = 512,
    ):
        # 0. default height and width
        num_frames = num_frames or (self.transformer.config.sample_size_t - 1) * self.vae.vae_scale_factor[0] + 1
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]
        video_size = (num_frames, height, width)
        # 1. Check inputs. Raise error if not correct
        prompts = (prompt, negative_prompt)
        embeds = (prompt_embeds, prompt_embeds_2, negative_prompt_embeds, negative_prompt_embeds_2)
        masks = (prompt_attention_mask, prompt_attention_mask_2, 
                 negative_prompt_attention_mask, negative_prompt_attention_mask_2)

        self._check_inputs(prompts, video_size, embeds, masks)
        self._guidance_scale = guidance_scale

        # 2. Define call parameters
        batch_size = self._get_batch(prompt, prompt_embeds)
        device = self.device

        # 3. Encode input prompt
        encode_kwarg = {"num_samples_per_prompt":num_samples_per_prompt,
                "do_classifier_free_guidance":self.do_classifier_free_guidance,
                "max_sequence_length":max_sequence_length}
        mask_emb_1 = (prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask) 
        mask_emb_2 = (prompt_embeds_2, negative_prompt_embeds_2, 
                      prompt_attention_mask_2, negative_prompt_attention_mask_2) 
        all_mask_emb = (mask_emb_1, mask_emb_2)
        # If sp,  the prompt_embeds the size [B, S/N, C]
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = self._get_embeding(prompts, all_mask_emb, encode_kwarg)


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        # if sp, the latent [B C [T//N] W H] 
        latents = self._get_latent(video_size, batch_size, num_samples_per_prompt, latents)

        # 8. Denoising loop
        all_guidance = (guidance_scale, guidance_rescale)
        input_embeds = prompt_embeds, prompt_embeds_2, prompt_attention_mask
        latents = self._sampling(timesteps, latents, input_embeds, all_guidance)

        if not output_type == "latent":
            videos = self._decode_latents(latents)
            videos = videos[:, :num_frames, :height, :width]
        else:
            videos = latents
        return (videos, )

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def _get_encode_kwarg(self, encode_kwarg):
        encode_kwarg = encode_kwarg or {}
        num_samples_per_prompt = encode_kwarg.get("num_samples_per_prompt", 1)
        do_classifier_free_guidance = encode_kwarg.get("do_classifier_free_guidance", True)
        max_sequence_length = encode_kwarg.get("max_sequence_length", None)
        return num_samples_per_prompt, do_classifier_free_guidance, max_sequence_length

    def _encode_prompt(
        self,
        prompts,
        mask_emb=None,
        encode_kwarg=None,
        text_encoder_index: int = 0
    ):  
        (num_samples_per_prompt, do_classifier_free_guidance, 
        max_sequence_length) = self._get_encode_kwarg(encode_kwarg)

        (prompt, negative_prompt) = prompts
        if mask_emb is None:
            mask_emb = [None] * 4
        (prompt_embeds, negative_prompt_embeds, 
        prompt_attention_mask, negative_prompt_attention_mask) = mask_emb 

        device = self.device
        dtype = self.transformer.dtype

        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]
        encoder = (tokenizers[text_encoder_index], text_encoders[text_encoder_index], text_encoder_index)

        max_length = self._get_length(max_sequence_length, text_encoder_index)
        batch_size = self._get_batch(prompt, prompt_embeds)
        encode_kwarg["max_sequence_length"] = max_length


        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._encode_prompt_process(
                prompt, encode_kwarg, encoder, trunc_test=True)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_samples_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_samples_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = self._standerd_neg_prompt(prompts, batch_size)
            negative_prompt_embeds, negative_prompt_attention_mask = self._encode_prompt_process(
                negative_prompt, encode_kwarg, encoder)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_samples_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_samples_per_prompt, seq_len, -1)
        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask
   
    def _get_length(self, max_sequence_length, text_encoder_index):
        if max_sequence_length is None:
            if text_encoder_index == 0:
                max_length = 512
            if text_encoder_index == 1:
                max_length = 77
        else:
            max_length = max_sequence_length
        return max_length
    
    def _get_batch(self, prompt, prompt_embeds):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        return batch_size

    def _encode_prompt_process(self, prompt, encode_kwarg, encoder, trunc_test=False):
        device = self.device
        tokenizer, text_encoder, text_encoder_index = encoder
        num_samples_per_prompt, _, max_length = self._get_encode_kwarg(encode_kwarg)
        text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        if trunc_test:
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                print("warning:", (
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                ))

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            attention_mask=prompt_attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        if text_encoder_index == 1:
            prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d for clip

        prompt_attention_mask = prompt_attention_mask.repeat(num_samples_per_prompt, 1)
        return prompt_embeds, prompt_attention_mask
    
    def _standerd_neg_prompt(self, prompts, batch_size):
        (prompt, negative_prompt) = prompts
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif prompt is not None and type(prompt) is not type(negative_prompt):
            raise ValueError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt
        return uncond_tokens

    def _sampling(self, timesteps, latents, input_embeds, all_guidance):
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = input_embeds
        guidance_scale, guidance_rescale, = all_guidance
        # ==================prepare my shape=====================================     
        # [B T W H] or [B T/N W H]        
        attention_mask = torch.ones_like(latents)[:, 0].repeat(2, 1, 1, 1).to(device=self.device)
        # If sp, recover attention_mask to the [B T W H]
        if get_sequence_parallel_state():
            attention_mask = attention_mask.repeat(1, get_sequence_parallel_size(), 1, 1)
        for step_id, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            scale_model = False
            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scale_model = True
            # Expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
            if isinstance(t, torch.Tensor):
                timestep = t.expand(latent_model_input.shape[0])
            else:
                timestep = torch.tensor([t] * latent_model_input.shape[0], device=self.device).to(
                    dtype=latent_model_input.dtype)
              
            noise_pred = self.transformer(
                latent_model_input,
                attention_mask=attention_mask, 
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                pooled_projections=prompt_embeds_2,
                step_id=step_id,
            )[0]
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and guidance_rescale > 0.0 and scale_model:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if get_sequence_parallel_state():
            world_size = get_sequence_parallel_size()
            latents_shape = list(latents.shape)  # b c t//sp h w
            full_shape = [latents_shape[0] * world_size] + latents_shape[1:]  # # b*sp c t//sp h w
            all_latents = torch.zeros(full_shape, dtype=latents.dtype, device=latents.device)
            torch.distributed.all_gather_into_tensor(all_latents, latents)
            latents_list = list(all_latents.chunk(world_size, dim=0))
            latents = torch.cat(latents_list, dim=2)
        return latents

    def _check_inputs(
        self,
        prompts,
        video_size,
        embeds,
        masks
    ):  
        num_frames, height, width = video_size
        # 1. Check inputs. Raise error if not correct
        suffix_2 = "_2"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        for i, prompt in enumerate(prompts):
            if prompt is not None and embeds[i] is not None:
                prefix = "" if i % 2 == 0 else "negative_"
                raise ValueError(
                    f"Cannot forward both `{prefix}prompt`: {prompt} and `{prefix}prompt_embeds`. Please make sure to"
                    " only forward one of the two.")
        for i in range(2):
            if prompts[0] is None and embeds[i] is None:
                suffix = "" if i % 2 == 0 else suffix_2 
                raise ValueError(
                    f"Provide either `prompt` or `prompt_embeds{suffix}`. "
                    f"Cannot leave both `prompt` and `prompt_embeds{suffix}` undefined.")

        if prompts[0] is not None and (not isinstance(prompts[0], str) and not isinstance(prompts[0], list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompts[0])}")

        #embed contain all the 
        for i, (emb, mask) in enumerate(zip(embeds, masks)):
            if emb is not None and mask is None:
                prefix = "" if i < 2 else "negative_"
                suffix = "" if i % 2 == 0 else suffix_2 
                raise ValueError(f"Must provide `{prefix}prompt_attention_mask{suffix}` "
                                        "when specifying `{prefix}prompt_embeds{suffix}`.")

        for i in range(2):
            if embeds[i] is not None and embeds[i + 2] is not None:
                suffix = "" if i % 2 == 0 else suffix_2 
                raise ValueError(
                    f"`prompt_embeds{suffix}` and `negative_prompt_embeds{suffix}` must have the same shape" 
                    f"when passed directly, but got: `prompt_embeds{suffix}` {embeds[i].shape} != "
                    f"`negative_prompt_embeds{suffix}` {embeds[i+2].shape}.")

    def _prepare_latents(self, batch_size, num_channels_latents, video_size, latents=None):
        num_frames, height, width = video_size
        shape = (
            batch_size,
            num_channels_latents,
            (int(num_frames) - 1) // self.vae.vae_scale_factor[0] + 1, 
            int(height) // self.vae.vae_scale_factor[1],
            int(width) // self.vae.vae_scale_factor[2],
        )
        device = self.device
        dtype = self.transformer.dtype

        if latents is None:
            latents = torch.randn(shape, dtype=dtype, device=device)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    def _get_embeding(self, prompts, all_mask_emb, encode_kwarg):
        device = self.device
        mask_emb_1, mask_emb_2 = all_mask_emb
        (prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        ) = self._encode_prompt(prompts, mask_emb_1, encode_kwarg, text_encoder_index=0)

        if self.tokenizer_2 is not None:
            (prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            ) = self._encode_prompt(prompts, mask_emb_2, encode_kwarg, text_encoder_index=1)
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_attention_mask_2 = None
            negative_prompt_attention_mask_2 = None

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            if self.tokenizer_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
                prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        if self.tokenizer_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(device=device)
            prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
        # If sp, split prompt_embeds to [B S/N C]
        if get_sequence_parallel_state():
            world_size = get_sequence_parallel_size()
            prompt_embeds = rearrange(prompt_embeds, 'b (n x) h -> b n x h',  
                n=world_size, x=prompt_embeds.shape[1] // world_size).contiguous()
            rank = get_sequence_parallel_rank()
            prompt_embeds = prompt_embeds[:, rank, :, :]

        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
        if prompt_attention_mask.ndim == 2:
            prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
        if prompt_embeds_2 is not None and prompt_embeds_2.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d
        return prompt_embeds, prompt_embeds_2, prompt_attention_mask

    def _get_latent(self, video_size, batch_size, num_samples_per_prompt, latents):

        (num_frames, height, width) = video_size
        world_size = get_sequence_parallel_size()
        num_channels_latents = self.transformer.config.in_channels
        video_size = (
            (num_frames + world_size - 1) // world_size, 
            height, width)

        latents = self._prepare_latents(
            batch_size * num_samples_per_prompt,
            num_channels_latents, video_size, latents)
        return latents
    
    def _decode_latents(self, latents):
        video = self.vae.decode(latents.to(self.vae.vae.dtype))
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8)
        video = video.cpu().permute(0, 1, 3, 4, 2).contiguous() # b t h w c
        return video