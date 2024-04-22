# Copyright 2023 Huawei Technologies Co., Ltd
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

import argparse
import csv
import json
import os
import time
from typing import Callable, List, Optional, Union
import numpy as np

import torch
import mindietorch
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, SASolverScheduler

from background_runtime_cache import BackgroundRuntime, RuntimeIOInfo
from mindietorch import _enums

clip_time = 0
unet_time = 0
vae_time = 0
p1_time = 0
p2_time = 0
p3_time = 0


class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            batch_size: int,
            num_images_per_prompt: int = 1,
            max_num_prompts: int = 0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)

        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration

        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret

    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break
                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))


class AIEStableDiffusionXLPipeline(StableDiffusionXLPipeline):

    def parser_args(self, args):
        self.args = args
        self.is_init = False
        if isinstance(self.args.device, list):
            self.device_0, self.device_1 = args.device
        else:
            self.device_0 = args.device

    def compile_aie_model(self):
        if self.is_init:
            return

        mindietorch.set_device(self.device_0)

        in_channels = self.unet.config.out_channels
        sample_size = self.unet.config.sample_size
        encoder_hidden_size_2 = self.text_encoder_2.config.hidden_size
        encoder_hidden_size = self.text_encoder.config.hidden_size + encoder_hidden_size_2
        max_position_embeddings = self.text_encoder.config.max_position_embeddings

        batch_size = self.args.batch_size

        if self.args.soc == "Duo":
            soc_version = "Ascend310P3"
        elif self.args.soc == "A2":
            soc_version = "Ascend910B4"
        else:
            print("unsupport soc_version, please check!")
            return

        vae_compiled_path = os.path.join(self.args.output_dir, f"vae/vae_bs{batch_size}_aie_compile.ts")
        if os.path.exists(vae_compiled_path):
            self.compiled_vae_model = torch.jit.load(vae_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"vae/vae_bs{batch_size}.pt")).eval()

            if batch_size <= 2:
                self.compiled_vae_model = (
                    mindietorch.compile(model,
                                        inputs=[
                                            mindietorch.Input((self.args.batch_size, in_channels,
                                                               sample_size, sample_size),
                                                              dtype=mindietorch.dtype.FLOAT)],
                                        allow_tensor_replace_int=True,
                                        require_full_compilation=True,
                                        truncate_long_and_double=True,
                                        soc_version=soc_version,
                                        precision_policy=_enums.PrecisionPolicy.FP16,
                                        optimization_level=0
                                        ))
            else:
                min_shape = (1, 4, 128, 128)
                max_shape = (32, 4, 128, 128)

                self.compiled_vae_model = (
                    mindietorch.compile(model,
                                        inputs=[
                                            mindietorch.Input(min_shape=min_shape,
                                                              max_shape=max_shape,
                                                              dtype=mindietorch.dtype.FLOAT)],
                                        allow_tensor_replace_int=True,
                                        require_full_compilation=True,
                                        truncate_long_and_double=True,
                                        soc_version=soc_version,
                                        precision_policy=_enums.PrecisionPolicy.FP16,
                                        optimization_level=0
                                        ))
            torch.jit.save(self.compiled_vae_model, vae_compiled_path)

        clip1_compiled_path = os.path.join(self.args.output_dir, f"clip/clip_bs{batch_size}_aie_compile.ts")
        if os.path.exists(clip1_compiled_path):
            self.compiled_clip_model = torch.jit.load(clip1_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"clip/clip_bs{batch_size}.pt")).eval()
            self.compiled_clip_model = (
                mindietorch.compile(model,
                                    inputs=[mindietorch.Input((self.args.batch_size,
                                                               max_position_embeddings),
                                                              dtype=mindietorch.dtype.INT64)],
                                    allow_tensor_replace_int=True,
                                    require_full_compilation=False,
                                    truncate_long_and_double=False,
                                    precision_policy=_enums.PrecisionPolicy.FP16,
                                    soc_version=soc_version,
                                    optimization_level=0))
            torch.jit.save(self.compiled_clip_model, clip1_compiled_path)

        clip2_compiled_path = os.path.join(self.args.output_dir, f"clip/clip2_bs{batch_size}_aie_compile.ts")
        if os.path.exists(clip2_compiled_path):
            self.compiled_clip_model_2 = torch.jit.load(clip2_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"clip/clip2_bs{batch_size}.pt")).eval()
            self.compiled_clip_model_2 = (
                mindietorch.compile(model,
                                    inputs=[mindietorch.Input((self.args.batch_size,
                                                               max_position_embeddings),
                                                              dtype=mindietorch.dtype.INT64)],
                                    allow_tensor_replace_int=True,
                                    require_full_compilation=False,
                                    truncate_long_and_double=False,
                                    min_block_size=1,
                                    precision_policy=_enums.PrecisionPolicy.FP16,
                                    soc_version=soc_version,
                                    optimization_level=0))
            torch.jit.save(self.compiled_clip_model_2, clip2_compiled_path)

        scheduler_compiled_path = os.path.join(self.args.output_dir, f"ddim/ddim{batch_size}_aie_compile.ts")
        if os.path.exists(scheduler_compiled_path):
            self.compiled_scheduler = torch.jit.load(scheduler_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"ddim/ddim{batch_size}.pt")).eval()

            self.compiled_scheduler = (
                mindietorch.compile(model,
                                    inputs=[mindietorch.Input((1,
                                                               in_channels, sample_size,
                                                               sample_size),
                                                              dtype=mindietorch.dtype.FLOAT),
                                            mindietorch.Input((1,
                                                               in_channels, sample_size,
                                                               sample_size),
                                                              dtype=mindietorch.dtype.FLOAT),
                                            mindietorch.Input((1,),
                                                              dtype=mindietorch.dtype.INT64),
                                            mindietorch.Input((1,
                                                               in_channels, sample_size,
                                                               sample_size),
                                                              dtype=mindietorch.dtype.FLOAT),
                                            mindietorch.Input((1,),
                                                              dtype=mindietorch.dtype.INT64)],
                                    allow_tensor_replace_int=True,
                                    require_full_compilation=True,
                                    truncate_long_and_double=False,
                                    precision_policy=_enums.PrecisionPolicy.FP16,
                                    soc_version=soc_version,
                                    optimization_level=0))
            torch.jit.save(self.compiled_scheduler, scheduler_compiled_path)

        if not self.args.use_cache:
            unet_compiled_path = os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_aie_compile.ts")
            if os.path.exists(unet_compiled_path):
                self.compiled_unet_model = torch.jit.load(unet_compiled_path).eval()
            else:
                model = torch.jit.load(os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}.pt")).eval()

                self.compiled_unet_model = (
                    mindietorch.compile(model,
                                        inputs=[mindietorch.Input((batch_size,
                                                                   in_channels, sample_size,
                                                                   sample_size),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((1,),
                                                                  dtype=mindietorch.dtype.INT64),
                                                mindietorch.Input((batch_size,
                                                                   max_position_embeddings,
                                                                   encoder_hidden_size),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((batch_size,
                                                                   encoder_hidden_size_2),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((batch_size, 6),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                ],
                                        allow_tensor_replace_int=True,
                                        require_full_compilation=True,
                                        truncate_long_and_double=True,
                                        soc_version=soc_version,
                                        precision_policy=_enums.PrecisionPolicy.FP16,
                                        optimization_level=0
                                        ))
                torch.jit.save(self.compiled_unet_model, unet_compiled_path)

        if self.args.use_cache:
            unet_skip_compiled_path = os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_aie_compile_1.ts")
            if os.path.exists(unet_skip_compiled_path):
                self.compiled_unet_model_skip = torch.jit.load(unet_skip_compiled_path).eval()
            else:
                model = torch.jit.load(os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_1.pt")).eval()

                self.compiled_unet_model_skip = (
                    mindietorch.compile(model,
                                        inputs=[mindietorch.Input((batch_size,
                                                                   in_channels, sample_size,
                                                                   sample_size),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((1,),
                                                                  dtype=mindietorch.dtype.INT64),
                                                mindietorch.Input((batch_size,
                                                                   max_position_embeddings,
                                                                   encoder_hidden_size),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((batch_size,
                                                                   encoder_hidden_size_2),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((batch_size, 6),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((1,),
                                                                  dtype=mindietorch.dtype.INT64),
                                                mindietorch.Input(
                                                    (batch_size, 1280, sample_size // 2, sample_size // 2),
                                                    dtype=mindietorch.dtype.FLOAT),
                                                ],
                                        allow_tensor_replace_int=True,
                                        require_full_compilation=True,
                                        truncate_long_and_double=True,
                                        soc_version=soc_version,
                                        precision_policy=_enums.PrecisionPolicy.FP16,
                                        optimization_level=0
                                        ))

                torch.jit.save(self.compiled_unet_model_skip, unet_skip_compiled_path)

            unet_cache_compiled_path = os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_aie_compile_0.ts")
            if os.path.exists(unet_cache_compiled_path):
                self.compiled_unet_model_cache = torch.jit.load(unet_cache_compiled_path).eval()
            else:
                model = torch.jit.load(os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_0.pt")).eval()
                self.compiled_unet_model_cache = (
                    mindietorch.compile(model,
                                        inputs=[mindietorch.Input((batch_size,
                                                                   in_channels, sample_size,
                                                                   sample_size),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((1,),
                                                                  dtype=mindietorch.dtype.INT64),
                                                mindietorch.Input((batch_size,
                                                                   max_position_embeddings,
                                                                   encoder_hidden_size),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((batch_size,
                                                                   encoder_hidden_size_2),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((batch_size, 6),
                                                                  dtype=mindietorch.dtype.FLOAT),
                                                mindietorch.Input((1,),
                                                                  dtype=mindietorch.dtype.INT64),
                                                ],
                                        allow_tensor_replace_int=True,
                                        require_full_compilation=False,
                                        truncate_long_and_double=False,
                                        soc_version=soc_version,
                                        precision_policy=_enums.PrecisionPolicy.FP16,
                                        optimization_level=0,
                                        ))
                torch.jit.save(self.compiled_unet_model_cache, unet_cache_compiled_path)

        runtime_info_cache = RuntimeIOInfo(
            input_shapes=[
                (batch_size, in_channels, sample_size, sample_size),
                (1,),
                (batch_size, max_position_embeddings, encoder_hidden_size),
                (batch_size, encoder_hidden_size_2),
                (batch_size, 6),
                (1,)
            ],
            input_dtypes=[np.float32, np.int64, np.float32, np.float32, np.float32, np.int64],  ########################
            output_shapes=[(batch_size, in_channels, sample_size, sample_size),
                           (batch_size, 1280, sample_size, sample_size)],
            output_dtypes=[np.float32, np.float32]
        )

        if hasattr(self, 'device_1'):
            self.unet_bg = BackgroundRuntime.clone(self.device_1, [unet_cache_compiled_path, unet_skip_compiled_path],
                                                   runtime_info_cache)
            self.use_parallel_inferencing = True

        self.is_init = True

    def encode_prompt(
            self,
            prompt,
            prompt_2,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            lora_scale,
            clip_skip
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.compiled_clip_model, self.compiled_clip_model_2] if self.compiled_clip_model is not None
            else [self.compiled_clip_model_2]
        )

        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        # flag = 0
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, tokenizer)
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])

            # We are only ALWAYS interested in the pooled output of the final text encoder

            global clip_time
            start = time.time()
            prompt_embeds_npu = text_encoder(text_input_ids.to(f'npu:{self.device_0}'))

            pooled_prompt_embeds = prompt_embeds_npu[0].to('cpu')
            clip_time += time.time() - start

            if clip_skip is None:
                prompt_embeds = prompt_embeds_npu[2][-2].to('cpu')

            else:
                # "2" because SDXL always indexes from the penultimate layer.????待定
                prompt_embeds = prompt_embeds_npu.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(f'npu:{self.device_0}'))[0].to('cpu')
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_prompt_embeds = [torch.from_numpy(text) for text in negative_prompt_embeds]
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device="cpu")
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device="cpu")
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def ascendie_infer(
            self,
            prompt: Union[str, List[str]],
            prompt_2: Optional[Union[str, List[str]]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[dict[str, any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[tuple[int, int]] = None,
            crops_coords_top_left: tuple[int, int] = (0, 0),
            target_size: Optional[tuple[int, int]] = None,
            negative_original_size: Optional[tuple[int, int]] = None,
            negative_crops_coords_top_left: tuple[int, int] = (0, 0),
            negative_target_size: Optional[tuple[int, int]] = None,
            clip_skip: Optional[int] = None,
            skip_steps=None,
            flag_ddim: int = None,
            flag_cache: int = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        """
        global p1_time, p2_time, p3_time
        start = time.time()

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt
        lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )

        p1_time += time.time() - start
        start1 = time.time()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        prompt_embeds = prompt_embeds.to(device)
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        negative_add_time_ids = negative_add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # 8.1 Apply denoising_end
        if (
                denoising_end is not None
                and isinstance(denoising_end, float)
                and denoising_end > 0
                and denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        global unet_time
        global vae_time

        cache = None
        skip_flag = torch.ones([1], dtype=torch.long)
        cache_flag = torch.zeros([1], dtype=torch.long)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if not self.use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(f'npu:{self.device_0}')
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if self.use_parallel_inferencing and do_classifier_free_guidance:
                self.unet_bg.infer_asyn([
                    latent_model_input.numpy(),  # [1, 4, 128, 128]
                    t[None].numpy().astype(np.int64),
                    prompt_embeds.to('cpu').numpy(),  # [1, 77, 2048]
                    add_text_embeds.to('cpu').numpy(),  # [1, 1280]
                    add_time_ids.numpy(),  # .astype(torch.long)
                    skip_flag.numpy(),
                ],
                    skip_steps[i])

            latent_model_input_npu = latent_model_input.to(f'npu:{self.device_0}')  # [1, 4, 128, 128]

            start = time.time()
            if skip_steps[i]:
                noise_pred_npu = self.compiled_unet_model_skip(latent_model_input_npu,  # [1, 4, 128, 128]
                                                               t.to(torch.int64)[None].to(f'npu:{self.device_0}'),
                                                               negative_prompt_embeds.to(f'npu:{self.device_0}'),
                                                               # [1, 77, 2048]
                                                               negative_pooled_prompt_embeds.to(f'npu:{self.device_0}'),
                                                               # [1, 1280]
                                                               negative_add_time_ids.to(f'npu:{self.device_0}'),
                                                               skip_flag.to(f'npu:{self.device_0}'),
                                                               cache, )  # if_skip, cache
                noise_pred = noise_pred_npu  # .to('cpu')
            else:
                start = time.time()
                outputs = self.compiled_unet_model_cache(latent_model_input_npu,
                                                         t.to(torch.int64)[None].to(f'npu:{self.device_0}'),
                                                         negative_prompt_embeds.to(f'npu:{self.device_0}'),
                                                         negative_pooled_prompt_embeds.to(f'npu:{self.device_0}'),
                                                         negative_add_time_ids.to(f'npu:{self.device_0}'),
                                                         cache_flag.to(f'npu:{self.device_0}'),
                                                         )
                cache = outputs[1]
                noise_pred = outputs[0]

            unet_time += time.time() - start

            # perform guidance
            if do_classifier_free_guidance:
                if self.use_parallel_inferencing:
                    if (skip_steps[i]):
                        noise_pred_text = torch.from_numpy(self.unet_bg.wait_and_get_outputs()[0])

                    else:
                        out = self.unet_bg.wait_and_get_outputs()  ##########################################
                        noise_pred_text = torch.from_numpy(out[0])

                else:
                    noise_pred, noise_pred_text = noise_pred.chunk(2)

                x = np.array(i, dtype=np.int64)
                y = torch.from_numpy(x).long()

                latents = self.compiled_scheduler(  # 2、分别输入两类噪声预测，就可以不用额外增加concat
                    noise_pred.to(f'npu:{self.device_0}'),  # 无条件
                    noise_pred_text.to(f'npu:{self.device_0}'),  # 有条件
                    t[None].to(f'npu:{self.device_0}'),
                    latents.to(f'npu:{self.device_0}'),
                    y[None].to(f'npu:{self.device_0}')).to('cpu')

        p2_time = time.time() - start1
        start3 = time.time()
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            latents = latents / self.vae.config.scaling_factor
            latents = self.vae.post_quant_conv(latents)
            start = time.time()
            image = self.compiled_vae_model(latents.to(f'npu:{self.device_0}')).to('cpu')
            # image = image.unsqueeze(0)
            vae_time += time.time() - start
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        p3_time += time.time() - start3
        return (image, None)


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [int(v) for v in value.split(',')]
        for ivalue in ilist[:2]:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError(
                    "{} of device:{} is invalid. valid value range is [{}, {}]"
                    .format(ivalue, value, min_value, max_value))
        return ilist[:2]
    else:
        # default as single int value
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError(
                "device:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, min_value, max_value))
        return ivalue


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./stable-diffusion-xl-base-1.0",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts.txt",
        help="A text file of prompts for generating images.",
    )
    parser.add_argument(
        "--prompt_file_type",
        choices=["plain", "parti"],
        default="plain",
        help="Type of prompt file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result images.",
    )
    parser.add_argument(
        "--info_file_save_path",
        type=str,
        default="./image_info.json",
        help="Path to save image information file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--device",
        type=check_device_range_valid,
        default=[0, 1],
        help="NPU device id. Give 2 ids to enable parallel inferencing.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="Number of images generated for each prompt.",
    )
    parser.add_argument(
        "--max_num_prompts",
        default=0,
        type=int,
        help="Limit the number of prompts (0: no limit).",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save compiled models.",
    )
    parser.add_argument(
        "--soc",
        choices=["Duo", "A2"],
        default="A2",
        help="soc_version.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["DDIM", "Euler", "DPM", "SA-Solver"],
        default="DDIM",
        help="Type of Sampling methods. Can choose from DDIM, Euler, DPM, SA-Solver",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cache during inference."
    )
    parser.add_argument(
        "--cache_steps",
        type=str,
        default="1,2,4,6,7,9,10,12,13,14,16,18,19,21,23,24,26,27,29,\
                30,31,33,34,36,37,39,40,42,43,45,47,48,49",  # 17+33
        help="Steps to use cache data."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pipe = AIEStableDiffusionXLPipeline.from_pretrained(args.model).to("cpu")

    flag_ddim = 0
    if args.scheduler == "DDIM":
        flag_ddim = 1
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "DPM":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "SA-Solver":
        pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config)

    pipe.parser_args(args)
    pipe.compile_aie_model()
    skip_steps = [0] * args.steps
    flag_cache = 0
    if args.use_cache:
        flag_cache = 1
        for i in args.cache_steps.split(','):
            if int(i) >= args.steps:
                continue
            skip_steps[int(i)] = 1

    use_time = 0
    prompt_loader = PromptLoader(args.prompt_file,
                                 args.prompt_file_type,
                                 args.batch_size,
                                 args.num_images_per_prompt,
                                 args.max_num_prompts)

    prompts_2 = ""
    infer_num = 0
    image_info = []
    current_prompt = None
    for i, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']

        print(f"[{infer_num + n_prompts}/{len(prompt_loader)}]: {prompts}")
        infer_num += args.batch_size

        start_time = time.time()
        images = pipe.ascendie_infer(
            prompts,
            prompts_2,
            num_inference_steps=args.steps,
            guidance_scale=5.0,  # 7.5,
            skip_steps=skip_steps,
            flag_ddim=flag_ddim,
            flag_cache=flag_cache,
        )
        use_time += time.time() - start_time

        for j in range(n_prompts):
            image_save_path = os.path.join(save_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)

    print(f"[info] infer number: {infer_num}; use time: {use_time:.3f}s\n"
          f"average time: {use_time / infer_num:.3f}s\n"
          f"clip time: {clip_time / infer_num:.3f}s\n"
          f"unet time: {unet_time / infer_num:.3f}s\n"
          f"vae time: {vae_time / infer_num:.3f}s\n"
          f"p1 time: {p1_time / infer_num:.3f}s\n"
          f"p2 time: {p2_time / infer_num:.3f}s\n"
          f"p3 time: {p3_time / infer_num:.3f}s\n")

    if hasattr(pipe, 'device_1'):
        if (pipe.unet_bg):
            pipe.unet_bg.stop()

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)

    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)
    mindietorch.finalize()


if __name__ == "__main__":
    main()