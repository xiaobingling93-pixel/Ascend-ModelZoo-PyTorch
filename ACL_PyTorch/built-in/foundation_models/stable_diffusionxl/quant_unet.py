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
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ais_bench.infer.interface import InferSession
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler
from modelslim.onnx.squant_ptq.onnx_quant_tools import OnnxCalibrator
from modelslim.onnx.squant_ptq.quant_config import QuantConfig
import numpy as np
import onnx
import torch

from background_session import BackgroundInferSession
from pipeline_ascend_stable_diffusionxl import AscendStableDiffusionXLPipeline
from stable_diffusionxl_ascend_infer import check_device_range_valid


class StableDiffusionXLDumpPipeline(AscendStableDiffusionXLPipeline):
    @torch.no_grad()
    def dump_data(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]],
        encode_session: InferSession,
        encode_session_2: InferSession,
        unet_sessions: List[List[InferSession]],
        scheduler_session: InferSession,
        dump_num: int = 10,
        use_npu_scheduler: bool = False,
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
    ):
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
            encode_session=encode_session,
            encode_session_2=encode_session_2
        )

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
        unet_session, unet_session_bg = unet_sessions
        use_parallel_inferencing = unet_session_bg is not None
        add_text_embeds = pooled_prompt_embeds

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
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

        if do_classifier_free_guidance and not use_parallel_inferencing:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_text_embeds = add_text_embeds.numpy()
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1).numpy()

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        prompt_embeds = prompt_embeds.numpy()
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

        dump_data = []
        start_id = num_inference_steps // 2 - dump_num // 2
        end_id = start_id + dump_num

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            t_numpy = t[None].numpy()
            if not use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if start_id <= i < end_id:
                dump_data.append([latent_model_input, t_numpy, prompt_embeds, add_text_embeds, add_time_ids])
            elif i == end_id:
                break

            if use_parallel_inferencing and do_classifier_free_guidance:
                unet_session_bg.infer_asyn(
                    [
                        latent_model_input,
                        t_numpy,
                        negative_prompt_embeds.numpy(),
                        negative_pooled_prompt_embeds.numpy(),
                        negative_add_time_ids.numpy(),
                    ],
                )

            inputs = [
                latent_model_input.numpy(),
                t_numpy.astype,
                prompt_embeds,
                add_text_embeds,
                add_time_ids,
            ]
            noise_pred = torch.from_numpy(unet_session.infer(inputs)[0])

            if do_classifier_free_guidance:
                if use_parallel_inferencing:
                    noise_pred_uncond = torch.from_numpy(unet_session_bg.wait_and_get_outputs()[0])
                else:
                    noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # perform guidance
            if use_npu_scheduler:
                latents = torch.from_numpy(
                    scheduler_session.infer(
                        [
                            noise_pred.numpy(),
                            t_numpy,
                            latents.numpy(),
                            np.array(i)
                        ]
                    )[0]
                )

            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False,
                )[0]

        return dump_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts.txt",
        help="A prompt file used to generate images.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Base path of om models.",
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="unet_quant", 
        help="Path to save result images.",
    )
    parser.add_argument(
        "--scheduler", 
        choices=["DDIM", "Euler", "DPM", "EulerAncestral", "DPM++SDEKarras"],
        default="DDIM", 
        help="Type of Sampling methods. Can choose from DDIM, Euler, DPM",
    )
    parser.add_argument(
        "--device", 
        type=check_device_range_valid, 
        default=0, 
        help="NPU device id. Give 2 ids to enable parallel inferencing."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50, 
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--data_num", 
        type=int, 
        default=10,
        help="the number of real data used in quant process"
    )
    parser.add_argument(
        "--data_free", 
        action='store_true', 
        help="do not use real data"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    unet_onnx = os.path.join(args.model_dir, "unet", "unet.onnx")

    if args.data_free:
        data = [[]]

    input_shape = ''
    model = onnx.load(unet_onnx)
    inputs = model.graph.input

    for inp in inputs:
        dims = inp.type.tensor_type.shape.dim
        shape = [str(x.dim_value) for x in dims]
        input_shape += inp.name + ':' + ','.join(shape) + ';'
        if args.data_free:
            dtype = inp.type.tensor_type.elem_type
            data_size = [x.dim_value for x in dims]
            if dtype == 1:
                data[0].append(np.random.random(data_size).astype(np.float32))
            if dtype == 7:
                data[0].append(np.random.randint(10, size=data_size).astype(np.int64))

    if not args.data_free:
        device = None
        device_2 = None

        if isinstance(args.device, list):
            device, device_2 = args.device
        else:
            device = args.device
        
        batch_size = inputs[0].type.tensor_type.shape.dim[0].dim_value
        if not device_2:
            batch_size = batch_size // 2

        pipe = StableDiffusionXLDumpPipeline.from_pretrained(args.model).to("cpu")

        use_npu_scheduler = False

        if args.scheduler == "DDIM":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            use_npu_scheduler = True

        elif args.scheduler == "Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == "DPM":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == "EulerAncestral":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif args.scheduler == "DPM++SDEKarras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.algorithm_type = 'sde-dpmsolver++'
            pipe.scheduler.config.use_karras_sigmas = True

        encoder_om = os.path.join(args.model_dir, "text_encoder", "text_encoder.om")
        encoder_om_2 = os.path.join(args.model_dir, "text_encoder", "text_encoder_2.om")
        unet_om = os.path.join(args.model_dir, "unet", "unet.om")

        encoder_session = InferSession(device, encoder_om)
        encoder_session_2 = InferSession(device, encoder_om_2)
        unet_session = InferSession(device, unet_om)

        if use_npu_scheduler:
            scheduler_om = os.path.join(args.model_dir, "ddim", "ddim.om")
            scheduler_session = InferSession(device, scheduler_om)
        else:
            scheduler_session = None

        unet_session_bg = None
        if device_2:
            unet_session_bg = BackgroundInferSession.clone(unet_session, device_2, [unet_om, ""])

        with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
            prompts = [line.strip() for line in f]

        data = pipe.dump_data(
            prompts[:batch_size],
            "",
            encoder_session,
            encoder_session_2,
            [unet_session, unet_session_bg],
            scheduler_session,
            args.data_num,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
            use_npu_scheduler=use_npu_scheduler,
        )

        if unet_session_bg:
            unet_session_bg.stop()
    
    config = QuantConfig(
        disable_names=[],
        quant_mode=0,
        amp_num=0,
        use_onnx=False,
        disable_first_layer=True,
        quant_param_ops=['Conv', 'MatMul'],
        atc_input_shape=input_shape[:-1],
        num_input=len(inputs),
    )

    calib = OnnxCalibrator(unet_onnx, config, calib_data=data)
    calib.run()
    quant_path = os.path.join(args.model_dir, args.save_path)
    if not os.path.exists(quant_path):
        os.makedirs(quant_path, mode=0o744)
    quant_onnx = os.path.join(quant_path, 'unet.onnx')
    calib.export_quant_onnx(quant_onnx, use_external=True)


if __name__ == "__main__":
    main()
