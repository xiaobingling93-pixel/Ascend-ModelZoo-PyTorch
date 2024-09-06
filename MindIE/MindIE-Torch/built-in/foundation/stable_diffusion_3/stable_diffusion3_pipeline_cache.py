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
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import mindietorch
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from stable_diffusion3_pipeline import PromptLoader, AIEStableDiffusion3Pipeline

tgate = 20
dit_time = 0
vae_time = 0
scheduler_time = 0
p1_time = 0
p2_time = 0
p3_time = 0


class AIEStableDiffusion3CachePipeline(AIEStableDiffusion3Pipeline):
    def compile_aie_model(self):
        if self.is_init:
            return
        size = self.args.batch_size
        batch_size = self.args.batch_size * 2
        tail = f"_{self.args.height}x{self.args.width}"

        vae_compiled_path = os.path.join(self.args.output_dir,
                                         f"vae/vae_bs{size}_compile{tail}.ts")
        self.compiled_vae_model = torch.jit.load(vae_compiled_path).eval()

        scheduler_compiled_path = os.path.join(self.args.output_dir,
                                               f"scheduler/scheduler_bs{size}_compile{tail}.ts")
        self.compiled_scheduler = torch.jit.load(scheduler_compiled_path).eval()

        clip1_compiled_path = os.path.join(self.args.output_dir,
                                           f"clip/clip_bs{size}_compile{tail}.ts")
        self.compiled_clip_model = torch.jit.load(clip1_compiled_path).eval()

        clip2_compiled_path = os.path.join(self.args.output_dir,
                                           f"clip/clip2_bs{size}_compile{tail}.ts")
        self.compiled_clip_model_2 = torch.jit.load(clip2_compiled_path).eval()

        t5_compiled_path = os.path.join(self.args.output_dir,
                                        f"clip/t5_bs{size}_compile{tail}.ts")
        self.compiled_t5_model = torch.jit.load(t5_compiled_path).eval()

        dit_cache_compiled_path = os.path.join(self.args.output_dir,
                                               f"dit/dit_bs{batch_size}_0_compile{tail}.ts")
        self.compiled_dit_cache_model = torch.jit.load(dit_cache_compiled_path).eval()

        if self.args.use_cache:
            dit_skip_compiled_path = os.path.join(self.args.output_dir,
                                                  f"dit/dit_bs{batch_size}_1_compile{tail}.ts")
            self.compiled_dit_skip_model = torch.jit.load(dit_skip_compiled_path).eval()

            dit_cache_end_compiled_path = os.path.join(self.args.output_dir,
                                                       f"dit/dit_bs{size}_0_compile{tail}.ts")
            self.compiled_dit_cache_end_model = torch.jit.load(dit_cache_end_compiled_path).eval()

            dit_skip_end_compiled_path = os.path.join(self.args.output_dir,
                                                      f"dit/dit_bs{size}_1_compile{tail}.ts")
            self.compiled_dit_skip_end_model = torch.jit.load(dit_skip_end_compiled_path).eval()

        self.is_init = True

    @torch.no_grad()
    def dit_infer(self, compiled_model, latent_model_input, prompt_embeds, pooled_prompt_embeds, timestep_npu,
                  cache_param, skip_flag, delta_cache, delta_cache_hidden):
        (noise_pred, delta_cache, delta_cache_hidden) = compiled_model(
            latent_model_input.to(f'npu:{self.device_0}'),
            prompt_embeds.to(f'npu:{self.device_0}'),
            pooled_prompt_embeds.to(f'npu:{self.device_0}'),
            timestep_npu,
            cache_param.to(f'npu:{self.device_0}'),
            skip_flag,
            delta_cache.to(f'npu:{self.device_0}'),
            delta_cache_hidden.to(f'npu:{self.device_0}'),
        )
        noise_pred = noise_pred.to("cpu")
        delta_cache = delta_cache.to("cpu")
        delta_cache_hidden = delta_cache_hidden.to("cpu")
        return (noise_pred, delta_cache, delta_cache_hidden)

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
            cache_param: torch.LongTensor = None,
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

        prompt_embeds_origin = prompt_embeds.clone()
        pooled_prompt_embeds_origin = pooled_prompt_embeds.clone()
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

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
        global scheduler_time

        skip_flag_true = torch.ones([1], dtype=torch.long).to(f'npu:{self.device_0}')
        skip_flag_false = torch.zeros([1], dtype=torch.long).to(f'npu:{self.device_0}')

        delta_cache = torch.zeros([2, 4096, 1536], dtype=torch.float32)
        delta_cache_hidden = torch.zeros([2, 154, 1536], dtype=torch.float32)

        cache_interval = cache_param[1]
        step_contrast = cache_param[3] % 2

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                start = time.time()
                timestep_npu = t.to(torch.int64)[None].to(f'npu:{self.device_0}')
                if not self.args.use_cache:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    (noise_pred, delta_cache, delta_cache_hidden) = self.dit_infer(
                        self.compiled_dit_cache_model,
                        latent_model_input,
                        prompt_embeds,
                        pooled_prompt_embeds,
                        timestep_npu, cache_param,
                        skip_flag_true, delta_cache,
                        delta_cache_hidden)
                else:
                    if i < tgate:
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    else:
                        if i == tgate:
                            _, delta_cache = delta_cache.chunk(2)
                            _, delta_cache_hidden = delta_cache_hidden.chunk(2)
                        latent_model_input = latents

                    if i < cache_param[3]:
                        (noise_pred, delta_cache, delta_cache_hidden) = self.dit_infer(
                            self.compiled_dit_cache_model,
                            latent_model_input,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            timestep_npu, cache_param,
                            skip_flag_true, delta_cache,
                            delta_cache_hidden)
                    else:
                        if i % cache_interval == step_contrast:
                            if i < tgate:
                                (noise_pred, delta_cache, delta_cache_hidden) = self.dit_infer(
                                    self.compiled_dit_cache_model,
                                    latent_model_input,
                                    prompt_embeds,
                                    pooled_prompt_embeds,
                                    timestep_npu, cache_param,
                                    skip_flag_true,
                                    delta_cache,
                                    delta_cache_hidden)
                            else:
                                (noise_pred, delta_cache, delta_cache_hidden) = self.dit_infer(
                                    self.compiled_dit_cache_end_model,
                                    latent_model_input,
                                    prompt_embeds_origin,
                                    pooled_prompt_embeds_origin,
                                    timestep_npu, cache_param,
                                    skip_flag_true,
                                    delta_cache,
                                    delta_cache_hidden)
                        else:
                            if i < tgate:
                                (noise_pred, delta_cache, delta_cache_hidden) = self.dit_infer(
                                    self.compiled_dit_skip_model,
                                    latent_model_input,
                                    prompt_embeds,
                                    pooled_prompt_embeds,
                                    timestep_npu, cache_param,
                                    skip_flag_false,
                                    delta_cache,
                                    delta_cache_hidden)
                            else:
                                (noise_pred, delta_cache, delta_cache_hidden) = self.dit_infer(
                                    self.compiled_dit_skip_end_model,
                                    latent_model_input,
                                    prompt_embeds_origin,
                                    pooled_prompt_embeds_origin,
                                    timestep_npu, cache_param,
                                    skip_flag_false,
                                    delta_cache,
                                    delta_cache_hidden)

                dit_time += (time.time() - start)

                # perform guidance
                if self.do_classifier_free_guidance and i < tgate:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                start = time.time()
                # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                step_index = torch.tensor(i).long()
                latents = self.compiled_scheduler(
                    noise_pred.to(f'npu:{self.device_0}'),
                    latents.to(f'npu:{self.device_0}'),
                    step_index[None].to(f'npu:{self.device_0}')
                ).to('cpu')
                scheduler_time += (time.time() - start)

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

        p2_time += time.time() - start1
        start2 = time.time()

        if output_type == "latent":
            image = latents
        else:
            start = time.time()
            image = self.compiled_vae_model(latents.to(f'npu:{self.device_0}')).to("cpu")
            vae_time += time.time() - start
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        p3_time += time.time() - start2

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
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cache during inference."
    )
    parser.add_argument(
        "--cache_param",
        default="1,2,20,10",
        type=str,
        help="steps to use cache data"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(args.device, list):
        mindietorch.set_device(args.device[0])
    else:
        mindietorch.set_device(args.device)
    pipe = AIEStableDiffusion3CachePipeline.from_pretrained(args.model).to("cpu")
    pipe.parser_args(args)
    pipe.compile_aie_model()

    cache_param = torch.zeros([4], dtype=torch.int64)
    cache_list = args.cache_param.split(',')
    cache_param[0] = int(cache_list[0])
    cache_param[1] = int(cache_list[1])
    cache_param[2] = int(cache_list[2])
    cache_param[3] = int(cache_list[3])
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
            guidance_scale=7.0,
            cache_param=cache_param
        )
        if i > 4:  # do not count the time spent inferring the first 0 to 4 images
            use_time += time.time() - start_time

        for j in range(n_prompts):
            image_save_path = os.path.join(save_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)

    print(
        f"[info] infer number: {infer_num - 5}; use time: {use_time:.3f}s\n"
        f"average time: {use_time / (infer_num - 5):.3f}s\n"
        f"dit time: {dit_time / infer_num:.3f}s\n"
        f"scheduler_time time: {scheduler_time / infer_num:.3f}s\n"
        f"vae time: {vae_time / infer_num:.3f}s\n"
        f"p1 time: {p1_time / infer_num:.3f}s\n"
        f"p2 time: {p2_time / infer_num:.3f}s\n"
        f"p3 time: {p3_time / infer_num:.3f}s\n"
    )

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)

    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)
    mindietorch.finalize()


if __name__ == "__main__":
    main()
