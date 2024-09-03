# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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
import logging
import csv
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
import mindietorch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from background_runtime import BackgroundRuntime, RuntimeIOInfo

clip_time = 0
t5_time = 0
dit_time = 0
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
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(max_num_prompts)
        else:
            print("This operation is not supported!")

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

    def load_prompts_hpsv2(self, max_num_prompts: int):
        with open('hpsv2_benchmark_prompts.json', 'r') as file:
            all_prompts = json.load(file)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                count += 1
                if max_num_prompts and count >= max_num_prompts:
                    break

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))


class AIEStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    def parser_args(self, args):
        self.args = args
        self.is_init = False
        if isinstance(self.args.device, list):
            self.device_0, self.device_1 = args.device
        else:
            self.device_0 = args.device
        self.data = None

    def compile_aie_model(self):
        if self.is_init:
            return
        size = self.args.batch_size
        if hasattr(self, 'device_1'):
            batch_size = self.args.batch_size
        else:
            batch_size = self.args.batch_size * 2

        tail = f"_{self.args.height}x{self.args.width}"
        vae_compiled_path = os.path.join(self.args.output_dir, f"vae/vae_bs{size}_compile{tail}.ts")
        self.compiled_vae_model = torch.jit.load(vae_compiled_path).eval()

        clip1_compiled_path = os.path.join(self.args.output_dir, f"clip/clip_bs{size}_compile{tail}.ts")
        self.compiled_clip_model = torch.jit.load(clip1_compiled_path).eval()

        clip2_compiled_path = os.path.join(self.args.output_dir, f"clip/clip2_bs{size}_compile{tail}.ts")
        self.compiled_clip_model_2 = torch.jit.load(clip2_compiled_path).eval()

        t5_compiled_path = os.path.join(self.args.output_dir, f"clip/t5_bs{size}_compile{tail}.ts")
        self.compiled_t5_model = torch.jit.load(t5_compiled_path).eval()

        dit_compiled_path = os.path.join(self.args.output_dir, f"dit/dit_bs{batch_size}_compile{tail}.ts")
        self.compiled_dit_model = torch.jit.load(dit_compiled_path).eval()

        self.use_parallel_inferencing = False

        if hasattr(self, 'device_1'):
            sample_size = self.transformer.config.sample_size
            in_channels = self.transformer.config.in_channels
            encoder_hidden_size_2 = self.text_encoder_2.config.hidden_size
            encoder_hidden_size = self.text_encoder.config.hidden_size + encoder_hidden_size_2
            max_position_embeddings = self.text_encoder.config.max_position_embeddings * 2

            runtime_info = RuntimeIOInfo(
                input_shapes=[
                    (batch_size, in_channels, sample_size, sample_size),
                    (batch_size, max_position_embeddings, encoder_hidden_size * 2),
                    (batch_size, encoder_hidden_size),
                    (1,),
                ],
                input_dtypes=[np.float32, np.float32, np.float32, np.int64],
                output_shapes=[(batch_size, in_channels, sample_size, sample_size)],
                output_dtypes=[np.float32]
            )
            self.dit_bg = BackgroundRuntime.clone(self.device_1, dit_compiled_path, runtime_info)
            self.use_parallel_inferencing = True

        self.is_init = True

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
    ):
        device = f"npu:{self.device_0}"
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logging.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        global t5_time
        start = time.time()
        prompt_embeds = self.compiled_t5_model(text_input_ids.to(device))[0].to('cpu')
        t5_time += (time.time() - start)

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device='cpu')
        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = f"npu:{self.device_0}"
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.compiled_clip_model, self.compiled_clip_model_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logging.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        global clip_time
        start = time.time()
        prompt_embeds = text_encoder(text_input_ids.to(device))
        pooled_prompt_embeds = prompt_embeds[0].to('cpu')
        clip_time += (time.time() - start)

        if clip_skip is None:
            prompt_embeds = prompt_embeds[2][-2].to('cpu')
        else:
            prompt_embeds = prompt_embeds[2][-(clip_skip + 2)].to('cpu')

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device='cpu')
        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

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

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3, num_images_per_prompt=num_images_per_prompt
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def forward(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            prompt_3: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            timesteps: List[int] = None,
            guidance_scale: float = 7.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        global p1_time, p2_time, p3_time
        start = time.time()
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
        )

        p1_time += (time.time() - start)
        start1 = time.time()

        if self.do_classifier_free_guidance and not self.use_parallel_inferencing:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            prompt_embeds, prompt_embeds_1 = negative_prompt_embeds, prompt_embeds
            pooled_prompt_embeds, pooled_prompt_embeds_1 = negative_pooled_prompt_embeds, pooled_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
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

        # 6. Denoising loop
        global dit_time
        global vae_time
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                if not self.use_parallel_inferencing and self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.to(torch.int64)[None].to(f"npu:{self.device_0}")
                else:
                    latent_model_input = latents
                    timestep = t.to(torch.int64)
                    self.dit_bg.infer_asyn([
                        latent_model_input.numpy(),
                        prompt_embeds_1.numpy(),
                        pooled_prompt_embeds_1.numpy(),
                        timestep[None].numpy().astype(np.int64)
                    ])
                    timestep_npu = timestep[None].to(f"npu:{self.device_0}")

                latent_model_input_npu = latent_model_input.to(f"npu:{self.device_0}")
                prompt_embeds_npu = prompt_embeds.to(f"npu:{self.device_0}")
                pooled_prompt_embeds_npu = pooled_prompt_embeds.to(f"npu:{self.device_0}")

                start = time.time()
                noise_pred = self.compiled_dit_model(
                    latent_model_input_npu,
                    prompt_embeds_npu,
                    pooled_prompt_embeds_npu,
                    timestep_npu
                ).to("cpu")
                dit_time += (time.time() - start)

                # perform guidance
                if self.do_classifier_free_guidance:
                    if self.use_parallel_inferencing:
                        noise_pred_text = torch.from_numpy(self.dit_bg.wait_and_get_outputs()[0])
                    else:
                        noise_pred, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred_text - noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        p2_time += (time.time() - start1)
        start2 = time.time()

        if output_type == "latent":
            image = latents
        else:
            start = time.time()
            image = self.compiled_vae_model(latents.to(f"npu:{self.device_0}")).to("cpu")
            vae_time += (time.time() - start)
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()
        p3_time += (time.time() - start2)

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


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
        default="./stable-diffusion-3-medium-diffusers",
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
        choices=["plain", "parti", "hpsv2"],
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
        default=28,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--device",
        type=check_device_range_valid,
        default=0,
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
        "--scheduler",
        choices=["FlowMatchEuler"],
        default="FlowMatchEuler",
        help="Type of Sampling methods. Default FlowMatchEuler",
    )
    parser.add_argument(
        "--height",
        default=1024,
        type=int,
        help="image height",
    )
    parser.add_argument(
        "--width",
        default=1024,
        type=int,
        help="image width"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pipe = AIEStableDiffusion3Pipeline.from_pretrained(args.model).to("cpu")
    pipe.parser_args(args)
    pipe.compile_aie_model()
    if isinstance(args.device, list):
        mindietorch.set_device(args.device[0])
    else:
        mindietorch.set_device(args.device)

    use_time = 0
    prompt_loader = PromptLoader(args.prompt_file,
                                 args.prompt_file_type,
                                 args.batch_size,
                                 args.num_images_per_prompt,
                                 args.max_num_prompts)

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
        images = pipe.forward(
            prompts,
            negative_prompt="",
            num_inference_steps=args.steps,
            guidance_scale=7.5,
        )
        if i > 4: # do not count the time spent inferring the first 0 to 4 images
            use_time += time.time() - start_time

        for j in range(n_prompts):
            image_save_path = os.path.join(save_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)

    infer_num = infer_num - 5 # do not count the time spent inferring the first 5 images
    print(f"[info] infer number: {infer_num}; use time: {use_time:.3f}s\n"
          f"average time: {use_time / infer_num:.3f}s\n")

    if hasattr(pipe, 'device_1'):
        if (pipe.dit_bg):
            pipe.dit_bg.stop()

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)

    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)
    mindietorch.finalize()


if __name__ == "__main__":
    main()
