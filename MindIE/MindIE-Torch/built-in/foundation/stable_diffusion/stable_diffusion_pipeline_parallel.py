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

import argparse
import csv
import json
import os
import time
from typing import Callable, List, Optional, Union
import numpy as np

import torch
import mindietorch
from mindietorch import _enums
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler, SASolverScheduler

from background_runtime import BackgroundRuntime, RuntimeIOInfo
from background_runtime_cache import BackgroundRuntimeCache, RuntimeIOInfoCache

clip_time = 0
unet_time = 0
vae_time = 0
p1_time = 0
p2_time = 0
p3_time = 0
scheduler_time = 0

class PromptLoader:
    def __init__(
            self,
            prompt_file: str,
            prompt_file_type: str,
            batch_size: int,
            num_images_per_prompt: int = 1,
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file)

        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file)

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

    def load_prompts_plain(self, file_path: str):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r", encoding='utf8') as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))


class AIEStableDiffusionPipeline(StableDiffusionPipeline):
    device_0 = None
    device_1 = None
    runtime = None
    engines = {}
    contexts = {}
    buffer_bindings = {}
    use_parallel_inferencing = False
    unet_bg = None
    unet_bg_cache = None

    def parser_args(self, args):
        self.args = args
        if isinstance(args.device, list):
            self.device_0, self.device_1 = args.device
            print(f'Using parallel inferencing on device {self.device_0} and {self.device_1}')
        else:
            self.device_0 = args.device
        self.is_init = False

    def compile_aie_model(self):
        if self.is_init:
            return

        mindietorch.set_device(self.device_0)
        in_channels = self.unet.config.out_channels
        sample_size = self.unet.config.sample_size
        encoder_hidden_size = self.text_encoder.config.hidden_size
        max_position_embeddings = self.text_encoder.config.max_position_embeddings

        batch_size = self.args.batch_size

        if self.args.soc == "Duo":
            soc_version = "Ascend310P3"
        elif self.args.soc == "A2":
            soc_version = "Ascend910B4"
        else:
            print("unsupport soc_version, please check!")
            return

        clip_compiled_path = os.path.join(self.args.output_dir, f"clip/clip_bs{batch_size}_aie_compile.ts")
        if os.path.exists(clip_compiled_path):
            self.compiled_clip_model = torch.jit.load(clip_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"clip/clip_bs{batch_size}.pt")).eval()

            self.compiled_clip_model = (
                mindietorch.compile(model,
                                  inputs=[mindietorch.Input((self.args.batch_size,
                                                           max_position_embeddings),
                                                          dtype=mindietorch.dtype.INT64)],
                                  allow_tensor_replace_int=True,
                                  require_full_compilation=True,
                                  truncate_long_and_double=True,
                                  precision_policy=_enums.PrecisionPolicy.FP16,
                                  soc_version=soc_version,
                                  optimization_level=0))
            torch.jit.save(self.compiled_clip_model, clip_compiled_path)

        vae_compiled_path = os.path.join(self.args.output_dir, f"vae/vae_bs{batch_size}_aie_compile.ts")
        if os.path.exists(vae_compiled_path):
            self.compiled_vae_model = torch.jit.load(vae_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"vae/vae_bs{batch_size}.pt")).eval()

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
            torch.jit.save(self.compiled_vae_model, vae_compiled_path)

        scheduler_compiled_path = os.path.join(self.args.output_dir, f"ddim/ddim{batch_size}_aie_compile.ts")
        if os.path.exists(scheduler_compiled_path):
            self.compiled_scheduler = torch.jit.load(scheduler_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"ddim/ddim{batch_size}.pt")).eval()

            self.compiled_scheduler = (
                mindietorch.compile(model,
                                  inputs=[mindietorch.Input((batch_size,
                                                           in_channels, sample_size,
                                                           sample_size),
                                                          dtype=mindietorch.dtype.FLOAT),
                                          mindietorch.Input((batch_size,
                                                           in_channels, sample_size,
                                                           sample_size),
                                                          dtype=mindietorch.dtype.FLOAT),
                                          mindietorch.Input((1,),
                                                          dtype=mindietorch.dtype.INT64),
                                          mindietorch.Input((batch_size,
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

        unet_compiled_path = os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_aie_compile.ts")
        if os.path.exists(unet_compiled_path):
            self.compiled_unet = torch.jit.load(unet_compiled_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}.pt")).eval()

            self.compiled_unet = (
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
                                                          dtype=mindietorch.dtype.FLOAT)],
                                  allow_tensor_replace_int=True,
                                  require_full_compilation=True,
                                  truncate_long_and_double=True,
                                  soc_version=soc_version,
                                  precision_policy=_enums.PrecisionPolicy.FP16,
                                  optimization_level=0
                                  ))
            torch.jit.save(self.compiled_unet, unet_compiled_path)
        
        runtime_info = RuntimeIOInfo(
            input_shapes=[
                (batch_size, in_channels, sample_size, sample_size),
                (1,),
                (batch_size, max_position_embeddings, encoder_hidden_size)
            ],
            input_dtypes=[np.float32, np.int64, np.float32],
            output_shapes=[(batch_size, in_channels, sample_size, sample_size)],
            output_dtypes=[np.float32]
        )
        if hasattr(self, 'device_1'):
            self.unet_bg = BackgroundRuntime.clone(self.device_1, unet_compiled_path, runtime_info)
            self.use_parallel_inferencing = True

        unet_cache_compiled_path = os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_aie_compile_0.ts")
        if os.path.exists(unet_cache_compiled_path):
            self.compiled_unet_cache = torch.jit.load(unet_cache_compiled_path).eval()
        else:
            unet_cache = torch.jit.load(os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_0.pt")).eval()

            self.compiled_unet_cache = (
                mindietorch.compile(unet_cache,
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
                                          mindietorch.Input((1,),
                                                          dtype=mindietorch.dtype.INT64)],
                                  allow_tensor_replace_int=True,
                                  require_full_compilation=False,
                                  truncate_long_and_double=True,
                                  soc_version=soc_version,
                                  precision_policy=_enums.PrecisionPolicy.FP16,
                                  optimization_level=0
                                  ))
            torch.jit.save(self.compiled_unet_cache, unet_cache_compiled_path)

        unet_skip_compiled_path = os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_aie_compile_1.ts")
        if os.path.exists(unet_skip_compiled_path):
            self.compiled_unet_skip = torch.jit.load(unet_skip_compiled_path).eval()
        else:
            unet_skip = torch.jit.load(os.path.join(self.args.output_dir, f"unet/unet_bs{batch_size}_1.pt")).eval()

            self.compiled_unet_skip = (
                mindietorch.compile(unet_skip,
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
                                          mindietorch.Input((1,),
                                                          dtype=mindietorch.dtype.INT64),
                                          mindietorch.Input((batch_size,
                                                           640, sample_size,
                                                           sample_size),
                                                          dtype=mindietorch.dtype.FLOAT)],
                                  allow_tensor_replace_int=True,
                                  require_full_compilation=True,
                                  truncate_long_and_double=True,
                                  soc_version=soc_version,
                                  precision_policy=_enums.PrecisionPolicy.FP16,
                                  optimization_level=0
                                  ))
            torch.jit.save(self.compiled_unet_skip, unet_skip_compiled_path)

        runtime_info_cache = RuntimeIOInfoCache(
            input_shapes=[
                (batch_size, in_channels, sample_size, sample_size),
                (1,),
                (batch_size, max_position_embeddings, encoder_hidden_size),
                (1,)
            ],
            input_dtypes=[np.float32, np.int64, np.float32, np.int64],
            output_shapes=[(batch_size, in_channels, sample_size, sample_size),
                           (batch_size, 640, sample_size, sample_size)],
            output_dtypes=[np.float32, np.float32]
        )

        runtime_info_skip = RuntimeIOInfoCache(
            input_shapes=[
                (batch_size, in_channels, sample_size, sample_size),
                (1,),
                (batch_size, max_position_embeddings, encoder_hidden_size),
                (1,)
            ],
            input_dtypes=[np.float32, np.int64, np.float32, np.int64, np.float32],
            output_shapes=[(batch_size, in_channels, sample_size, sample_size)],
            output_dtypes=[np.float32]
        )

        if hasattr(self, 'device_1'):
            self.unet_bg_cache = BackgroundRuntimeCache.clone(self.device_1, [unet_cache_compiled_path, unet_skip_compiled_path], runtime_info_cache)
            self.use_parallel_inferencing = True

        self.is_init = True

    @torch.no_grad()
    def ascendie_infer_ddim(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor],
            None]] = None,
            callback_steps: Optional[int] = 1,
            skip_steps = None,
            flag_cache: int = None,
            **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (畏) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        global p1_time, p2_time, p3_time, scheduler_time
        start1 = time.time()
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # check compile
        if not self.is_init:
            self.compile_aie_model()

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)

        text_embeddings_dtype = text_embeddings.dtype
        p1_time += time.time() - start1
        start2 = time.time()
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt,
                                       num_channels_latents, height, width,
                                       text_embeddings_dtype, device,
                                       generator, latents)

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        global unet_time
        global vae_time
        if self.use_parallel_inferencing and do_classifier_free_guidance:
            # Split embeddings
            text_embeddings, text_embeddings_2 = text_embeddings.chunk(2)

        cache = None
        skip_flag = torch.ones([1], dtype=torch.long)
        cache_flag = torch.zeros([1], dtype=torch.long)

        stream = mindietorch.npu.Stream(f'npu:{self.device_0}')
        for i, t in enumerate(self.progress_bar(timesteps)):
            if i == 50:
                break
            # expand the latents if we are doing classifier free guidance
            if not self.use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if self.use_parallel_inferencing and do_classifier_free_guidance:
                if flag_cache:
                    self.unet_bg_cache.infer_asyn([
                        latent_model_input.numpy(),
                        t[None].numpy().astype(np.int64),
                        text_embeddings_2.numpy(),
                        skip_flag.numpy(),
                        # cache_numpy,
                    ],
                    skip_steps[i])
                else:
                    self.unet_bg.infer_asyn([
                        latent_model_input.numpy(),
                        t[None].numpy().astype(np.int64),
                        text_embeddings_2.numpy(),
                    ])

            latent_model_input_npu = latent_model_input.to(f'npu:{self.device_0}')
            t_npu = t[None].to(f'npu:{self.device_0}')
            text_embeddings_npu = text_embeddings.to(f'npu:{self.device_0}')

            start = time.time()

            if flag_cache:
                with mindietorch.npu.stream(stream):
                    if (skip_steps[i]):
                        noise_pred = self.compiled_unet_skip(latent_model_input_npu,
                                                                t_npu,
                                                                text_embeddings_npu,
                                                                skip_flag.to(f'npu:{self.device_0}'),
                                                                cache)
                    else:
                        outputs = self.compiled_unet_cache(latent_model_input_npu,
                                                            t_npu,
                                                            text_embeddings_npu,
                                                            cache_flag.to(f'npu:{self.device_0}'),
                                                            )
                        noise_pred = outputs[0]
                        cache = outputs[1]
                    stream.synchronize()
            else:
                with mindietorch.npu.stream(stream):
                    noise_pred = self.compiled_unet(latent_model_input_npu, t_npu, text_embeddings_npu)
                    stream.synchronize()

            unet_time += time.time() - start

            # perform guidance
            # compute the previous noisy sample x_t -> x_t-1
            start = time.time()
            if do_classifier_free_guidance:
                if self.use_parallel_inferencing:
                    if flag_cache:
                        if (skip_steps[i]):
                            noise_pred_text = torch.from_numpy(self.unet_bg_cache.wait_and_get_outputs()[0])
                        else:
                            out = self.unet_bg_cache.wait_and_get_outputs()
                            noise_pred_text = torch.from_numpy(out[0])
                    else:
                        noise_pred_text = torch.from_numpy(self.unet_bg.wait_and_get_outputs()[0])
                else:
                    noise_pred, noise_pred_text = noise_pred.chunk(2)

                x = np.array(i, dtype=np.int64)
                y = torch.from_numpy(x).long()

                latents = self.compiled_scheduler(
                                noise_pred.to(f'npu:{self.device_0}'),
                                noise_pred_text.to(f'npu:{self.device_0}'),
                                t[None].to(f'npu:{self.device_0}'),
                                latents.to(f'npu:{self.device_0}'),
                                y[None].to(f'npu:{self.device_0}')).to('cpu')

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            scheduler_time += time.time() - start

        # 8. Post-processing
        p2_time += time.time() - start2
        start3 = time.time()

        # run inference
        start = time.time()
        image = self.compiled_vae_model(latents.to(f'npu:{self.device_0}')).to('cpu')
        vae_time += time.time() - start

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.clamp(0, 1).float().numpy()

        # 9. Run safety checker
        has_nsfw_concept = False

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        p3_time += time.time() - start3
        return (image, has_nsfw_concept)


    def ascendie_infer(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor],
            None]] = None,
            callback_steps: Optional[int] = 1,
            skip_steps = None,
            flag_cache: int = None,
            **kwargs,
    ):
        # 0. Default height and width to unet
        global p1_time, p2_time, p3_time, scheduler_time
        start1 = time.time()
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # check compile
        if not self.is_init:
            self.compile_aie_model()

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)

        text_embeddings_dtype = text_embeddings.dtype
        p1_time += time.time() - start1
        start2 = time.time()
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt,
                                       num_channels_latents, height, width,
                                       text_embeddings_dtype, device,
                                       generator, latents)

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        global unet_time
        global vae_time
        if self.use_parallel_inferencing and do_classifier_free_guidance:
            # Split embeddings
            text_embeddings, text_embeddings_2 = text_embeddings.chunk(2)

        cache = None
        skip_flag = torch.ones([1], dtype=torch.long)
        cache_flag = torch.zeros([1], dtype=torch.long)

        for i, t in enumerate(self.progress_bar(timesteps)):
            if i == 50:
                break

            # expand the latents if we are doing classifier free guidance
            if not self.use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if self.use_parallel_inferencing and do_classifier_free_guidance:
                if flag_cache:
                    self.unet_bg_cache.infer_asyn([
                        latent_model_input.numpy(),
                        t[None].numpy().astype(np.int64),
                        text_embeddings_2.numpy(),
                        skip_flag.numpy(),
                    ],
                    skip_steps[i])
                else:
                    self.unet_bg.infer_asyn([
                        latent_model_input.numpy(),
                        t[None].numpy().astype(np.int64),
                        text_embeddings_2.numpy(),
                    ])

            latent_model_input_npu = latent_model_input.to(f'npu:{self.device_0}')
            t_npu = t[None].to(f'npu:{self.device_0}')
            text_embeddings_npu = text_embeddings.to(f'npu:{self.device_0}')

            start = time.time()

            if flag_cache:
                if skip_steps[i]:
                    noise_pred = self.compiled_unet_skip(latent_model_input_npu,
                                                        t_npu,
                                                        text_embeddings_npu,
                                                        skip_flag.to(f'npu:{self.device_0}'),
                                                        cache).to('cpu')
                else:
                    outputs = self.compiled_unet_cache(latent_model_input_npu,
                                                    t_npu,
                                                    text_embeddings_npu,
                                                    cache_flag.to(f'npu:{self.device_0}'),
                                                    )
                    noise_pred = outputs[0].to('cpu')
                    cache = outputs[1]
            else:
                noise_pred = self.compiled_unet(latent_model_input_npu,
                                                t_npu,
                                                text_embeddings_npu).to('cpu')

            unet_time += time.time() - start

            # perform guidance
            start = time.time()
            if do_classifier_free_guidance:
                if self.use_parallel_inferencing:
                    if flag_cache:
                        if (skip_steps[i]):
                            noise_pred_text = torch.from_numpy(self.unet_bg_cache.wait_and_get_outputs()[0])
                        else:
                            out = self.unet_bg_cache.wait_and_get_outputs()
                            noise_pred_text = torch.from_numpy(out[0])
                    else:
                        noise_pred_text = torch.from_numpy(self.unet_bg.wait_and_get_outputs()[0])
                else:
                    noise_pred, noise_pred_text = noise_pred.chunk(2)

                noise_pred = noise_pred + guidance_scale * (noise_pred_text -
                                                            noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            scheduler_time += time.time() - start

        # 8. Post-processing
        p2_time += time.time() - start2
        start3 = time.time()

        # run inference
        start = time.time()
        image = self.compiled_vae_model(latents.to(f'npu:{self.device_0}')).to('cpu')
        vae_time += time.time() - start

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.clamp(0, 1).float().numpy()

        # 9. Run safety checker
        has_nsfw_concept = False

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        p3_time += time.time() - start3
        return (image, has_nsfw_concept)

    def _encode_prompt(
            self,
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt,
                                         padding="max_length",
                                         return_tensors="pt").input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
            print("[warning] The following part of your input was truncated"
                  " because CLIP can only handle sequences up to"
                  f" {self.tokenizer.model_max_length} tokens: {removed_text}")

        # run inference
        self.text_encoder.eval()
        global clip_time
        start = time.time()
        text_embeddings = self.compiled_clip_model(text_input_ids.to(f'npu:{self.device_0}')).to('cpu')
        clip_time += time.time() - start

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(uncond_tokens,
                                          padding="max_length",
                                          max_length=max_length,
                                          truncation=True,
                                          return_tensors="pt")

            # run inference
            start = time.time()
            uncond_embeddings = self.compiled_clip_model(uncond_input.input_ids.to(f'npu:{self.device_0}')).to('cpu')
            clip_time += time.time() - start

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings


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
        default="./stable-diffusion-2-1-base",
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
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "--scheduler",
        choices=["DDIM", "Euler", "DPM", "SA-Solver"],
        default="DDIM",
        help="Type of Sampling methods. Can choose from DDIM, Euler, DPM, SA-Solver",
    )
    parser.add_argument(
        "--soc",
        choices=["Duo", "A2"],
        default="Duo",
        help="soc_version.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save compiled models.",
    )
    parser.add_argument(
        "--use_cache", 
        action="store_true",
        help="Use cache during inference."
    )
    parser.add_argument(
        "--cache_steps", 
        type=str, 
        default="1,2,3,4,5,7,9,10,12,13,14,16,18,19,21,23,24,26,27,29,\
                30,31,33,34,36,37,39,40,41,43,44,45,47,48,49", 
        help="Steps to use cache data."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pipe = AIEStableDiffusionPipeline.from_pretrained(args.model).to("cpu")
    pipe.parser_args(args)
    if args.scheduler == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "DPM":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "SA-Solver":
        pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config)
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
                                 args.num_images_per_prompt)

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
        if args.scheduler == "DDIM":
            images = pipe.ascendie_infer_ddim(
                prompts,
                num_inference_steps=args.steps,
                skip_steps=skip_steps,
                flag_cache=flag_cache,
            )
        else:
            images = pipe.ascendie_infer(
                prompts,
                num_inference_steps=args.steps,
                skip_steps=skip_steps,
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
          )
    if hasattr(pipe, 'device_1'):
        if (pipe.unet_bg):
            pipe.unet_bg.stop()

        if (pipe.unet_bg_cache):
            pipe.unet_bg_cache.stop()

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)

    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)

    mindietorch.finalize()


if __name__ == "__main__":
    main()
