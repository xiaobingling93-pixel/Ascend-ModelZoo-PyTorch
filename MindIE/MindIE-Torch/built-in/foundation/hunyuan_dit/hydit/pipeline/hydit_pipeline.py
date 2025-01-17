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


from typing import List, Union, Tuple
import logging

import torch
from tqdm import tqdm
import numpy as np

from ..layers import RotaryPositionEmbedding
from ..utils import postprocess_pil, randn_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKENIZER_MAX_LENGTH = 256
MAX_PROMPT_LENGTH = 1024
NEGATIVE_PROMPT = '错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，'
STANDARD_RATIO = np.array(
    [
        1.0,  # 1:1
        4.0 / 3.0,  # 4:3
        3.0 / 4.0,  # 3:4
        16.0 / 9.0,  # 16:9
        9.0 / 16.0,  # 9:16
    ]
)
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],  # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],  # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],  # 3:4
    [(1280, 768)],  # 16:9
    [(768, 1280)],  # 9:16
]
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]
SUPPORTED_SHAPE = [
    (1024, 1024),
    (1280, 1280),  # 1:1
    (1024, 768),
    (1152, 864),
    (1280, 960),  # 4:3
    (768, 1024),
    (864, 1152),
    (960, 1280),  # 3:4
    (1280, 768),  # 16:9
    (768, 1280),  # 9:16
]


class HunyuanDiTPipeline:

    def __init__(
        self,
        scheduler,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        vae,
        args,
        input_size: Tuple[int, int] = (1024, 1024)
    ):
        super().__init__()
        torch.set_grad_enabled(False)

        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.transformer = transformer
        self.vae = vae
        self.input_size = input_size
        self._check_init_input()

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.hidden_states_batch = 2
        self.device = torch.device("npu")
        self.guidance_scale = args.guidance_scale

        # Set image height and width.
        height = self.input_size[0]
        width = self.input_size[1]
        self.height = int((height // 16) * 16)
        self.width = int((width // 16) * 16)
        if (self.height, self.width) not in SUPPORTED_SHAPE:
            width, height = map_to_standard_shapes(self.width, self.height)
            self.height = int(height)
            self.width = int(width)
            logger.warning(f"Reshaped to ({self.height}, {self.width}), Supported shapes are {SUPPORTED_SHAPE}")

        # Create image rotary position embedding
        self.rotary_pos_emb = self._get_rotary_pos_emb()

        # Only for hydit <= 1.1
        self.image_meta_size, self.style = self._get_v1_params(args)

        # Use DiT Cache
        self.use_cache = args.use_cache
        if self.use_cache:
            self.step_start = args.step_start
            self.step_interval = args.step_interval
            self.block_start = args.block_start
            self.num_blocks = args.num_blocks
            self.step_contrast = 9 % 2
            self.skip_flag_true = torch.ones([1], dtype=torch.int64).to(self.device)
            self.skip_flag_false = torch.zeros([1], dtype=torch.int64).to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 100,
        seed_generator: torch.Generator = None
    ):
        # 1. Check inputs. Raise error if not correct
        check_call_input(prompt, num_images_per_prompt, num_inference_steps, seed_generator)

        # 2. Define prompt and negative_prompt
        if prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
        negative_prompt = NEGATIVE_PROMPT
        if prompt is not None and not isinstance(prompt, type(negative_prompt)):
            raise ValueError(
                f"negative_prompt should be the same type to prompt, "
                f"but got {type(negative_prompt)} != {type(prompt)}."
            )
        prompt_info = (prompt, negative_prompt, num_images_per_prompt)

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, attention_mask, uncond_attention_mask = \
            self._encode_prompt(prompt_info, batch_size, embedder_t5=False)
        prompt_embeds_t5, negative_prompt_embeds_t5, attention_mask_t5, uncond_attention_mask_t5 = \
            self._encode_prompt(prompt_info, batch_size, embedder_t5=True)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        attention_mask = torch.cat([uncond_attention_mask, attention_mask])
        prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, prompt_embeds_t5])
        attention_mask_t5 = torch.cat([uncond_attention_mask_t5, attention_mask_t5])
        transformer_input = (attention_mask, prompt_embeds_t5, attention_mask_t5, self.image_meta_size, self.style)
        torch.npu.empty_cache()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        step = (timesteps, num_inference_steps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        shape = (batch_size * num_images_per_prompt,
                 num_channels_latents,
                 self.height // self.vae_scale_factor,
                 self.width // self.vae_scale_factor)
        latents = randn_tensor(shape, generator=seed_generator, device=self.device, dtype=prompt_embeds.dtype) * 1.0

        # 6. Denoising loop
        latents = self._sampling(latents, step, prompt_embeds, transformer_input, seed_generator)
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = postprocess_pil(image)

        return (image, None)


    def _check_init_input(self):
        if not isinstance(self.input_size, tuple):
            raise ValueError(f"The type of input_size must be tuple, but got {type(self.input_size)}.")
        if len(self.input_size) != 2:
            raise ValueError(f"The length of input_size must be 2, but got {len(self.input_size)}.")
        if self.input_size[0] % 8 != 0 or self.input_size[0] <= 0:
            raise ValueError(
                f"The height of input_size must be divisible by 8 and greater than 0, but got {self.input_size[0]}.")
        if self.input_size[1] % 8 != 0 or self.input_size[1] <= 0:
            raise ValueError(
                f"The width of input_size must be divisible by 8 and greater than 0, but got {self.input_size[1]}.")


    def _get_rotary_pos_emb(self):
        grid_height = self.height // 8 // self.transformer.config.patch_size
        grid_width = self.width // 8 // self.transformer.config.patch_size
        base_size = 512 // 8 // self.transformer.config.patch_size
        head_dim = self.transformer.config.hidden_size // self.transformer.config.num_heads

        rope = RotaryPositionEmbedding(head_dim)
        freqs_cis_img = rope.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
        if isinstance(freqs_cis_img, tuple) and len(freqs_cis_img) == 2:
            return (freqs_cis_img[0].to(self.device), freqs_cis_img[1].to(self.device))
        else:
            raise ValueError(f"The type of rotary_pos_emb must be tuple and the length must be 2.")


    def _get_v1_params(self, args):
        if args.use_style_cond and args.size_cond is not None:
            src_size_cond = args.size_cond
            if isinstance(src_size_cond, int):
                src_size_cond = [src_size_cond, src_size_cond]
            if not isinstance(src_size_cond, (list, tuple)):
                raise TypeError(f"The src_size_cond must be a list or tuple, but got {type(src_size_cond)}.")
            if len(src_size_cond) != 2:
                raise ValueError(f"The src_size_cond must be a tuple of 2 integers, but got {len(src_size_cond)}.")
            size_cond = list(src_size_cond) + [self.width, self.height, 0, 0]
            image_meta_size = torch.as_tensor([size_cond] * 2 * args.batch_size, device=args.device)
            style = torch.as_tensor([0, 0] * args.batch_size, device=args.device)
        else:
            image_meta_size = None
            style = None
        return image_meta_size, style


    def _encode_prompt(self, prompt_info, batch_size, embedder_t5=False):
        if not embedder_t5:
            text_encoder = self.text_encoder
            tokenizer = self.tokenizer
            max_length = self.tokenizer.model_max_length
        else:
            text_encoder = self.text_encoder_2
            tokenizer = self.tokenizer_2
            max_length = TOKENIZER_MAX_LENGTH

        prompt, negative_prompt, num_images_per_prompt = prompt_info
        # prompt_embeds
        prompt_embeds, attention_mask = self._encode_embeds(
            prompt, tokenizer, text_encoder, max_length, num_images_per_prompt)
        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        # negative_prompt_embeds
        negative_prompt_embeds, uncond_attention_mask = self._encode_negative_embeds(
            negative_prompt, tokenizer, text_encoder, prompt_embeds, num_images_per_prompt)
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, attention_mask, uncond_attention_mask


    def _encode_embeds(self, prompt, tokenizer, text_encoder, max_length, num_images_per_prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        attention_mask = text_inputs.attention_mask.to(self.device)
        prompt_embeds = text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]
        attention_mask = attention_mask.repeat(num_images_per_prompt, 1)

        return prompt_embeds, attention_mask


    def _encode_negative_embeds(self, negative_prompt, tokenizer, text_encoder, prompt_embeds, num_images_per_prompt):
        uncond_tokens: List[str]
        if isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        else:
            uncond_tokens = negative_prompt

        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        uncond_attention_mask = uncond_input.attention_mask.to(self.device)
        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=uncond_attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        uncond_attention_mask = uncond_attention_mask.repeat(num_images_per_prompt, 1)

        return negative_prompt_embeds, uncond_attention_mask


    def _sampling(self, latents, step, prompt_embeds, transformer_input, seed_generator):

        timesteps, num_inference_steps = step

        if self.use_cache:
            delta_cache = torch.zeros([2, 3840, 1408], dtype=torch.float16).to(self.device)
            step_start = self.step_start

        num_warmup_steps = len(timesteps) - num_inference_steps
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * self.hidden_states_batch)
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=latent_model_input.device)

                # if use_fp16
                latent_model_input = latent_model_input.half()
                t_expand = t_expand.half()
                prompt_embeds = prompt_embeds.half()

                # predict the noise residual
                tensor_input = (latent_model_input, t_expand, prompt_embeds, transformer_input, self.rotary_pos_emb)
                if not self.use_cache:
                    noise_pred = self.transformer(tensor_input)
                else:
                    cache_params = (self.block_start, self.num_blocks, delta_cache.half())
                    inputs = [tensor_input, self.use_cache, cache_params, self.skip_flag_false]
                    if i < step_start:
                        noise_pred, delta_cache = self.transformer(*inputs)
                    else:
                        if i % self.step_interval == self.step_contrast:
                            noise_pred, delta_cache = self.transformer(*inputs)
                        else:
                            inputs[-1] = self.skip_flag_true
                            noise_pred, delta_cache = self.transformer(*inputs)

                # if learn_sigma
                noise_pred, _ = noise_pred.chunk(2, dim=1)
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, seed_generator)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()

        return latents


    def _progress_bar(self, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(f"_progress_bar_config should be dict, but is {type(self._progress_bar_config)}.")

        if total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("total has to be defined.")


def check_call_input(prompt, num_images_per_prompt, num_inference_steps, seed_generator):
    if not isinstance(prompt, str):
        raise ValueError("The input prompt type must be strings.")
    if len(prompt) == 0 or len(prompt) >= MAX_PROMPT_LENGTH:
        raise ValueError(
            f"The length of the prompt should be (0, {MAX_PROMPT_LENGTH}), but got {len(prompt)}.")    
    if not isinstance(num_images_per_prompt, int):
        raise ValueError("The input num_images_per_prompt type must be an instance of int.")
    if num_images_per_prompt < 0:
        raise ValueError(
            f"Input num_images_per_prompt should be a non-negative integer, but got {num_images_per_prompt}.")
    if not isinstance(num_inference_steps, int):
        raise ValueError("The input num_inference_steps type must be an instance of int.")
    if num_inference_steps < 0:
        raise ValueError(
            f"Input num_inference_steps should be a non-negative integer, but got {num_inference_steps}.")
    if not isinstance(seed_generator, torch.Generator):
        raise ValueError(
            f"The type of input seed_generator must be torch.Generator, but got {type(seed_generator)}.")


def map_to_standard_shapes(target_width, target_height):
    target_ratio = target_width / target_height
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    return width, height