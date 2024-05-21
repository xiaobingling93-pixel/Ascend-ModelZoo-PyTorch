# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import numpy as np
import torch
import mindietorch
from mindietorch import _enums
import pickle
from typing import Callable, Dict, List, Optional, Union

from diffusers import StableVideoDiffusionPipeline
import diffusers.models.transformer_temporal
from diffusers.utils import load_image, export_to_video
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing,_compute_padding,_filter2d,_gaussian,_gaussian_blur2d,_append_dims,inspect,tensor2vid,StableVideoDiffusionPipelineOutput
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor

from background_runtime import BackgroundRuntime, RuntimeIOInfo

image_embed_time = 0
heightS = 512
widthS = 512
num_framesS = 25
Dshape = False

print("height:{},width:{},num_frames:{},vae_decode dynamic shape:{}".format(heightS,widthS,num_framesS,Dshape))

class AIEStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    device_0 = None
    device_1 = None
    runtime = None
    use_parallel_inferencing = False
    unet_bg = None

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

        in_channels = self.unet.config.in_channels
        batch_size = self.args.batch_size
        num_videos_per_prompt = 1
        height = heightS
        width = widthS
        num_frames = num_framesS if num_framesS is not None else self.unet.config.num_frames
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        seq_len = 1
        vae_encode_out=1024
        decode_chunk_size=self.args.decode_chunk_size
        num_inference_steps=self.args.num_inference_steps
        res=num_frames%decode_chunk_size
        channels_latents=in_channels // 2

        image_encoder_embed_path = os.path.join(self.args.output_dir, "image_encoder_embed/image_encoder_embed.ts")
        if os.path.exists(image_encoder_embed_path):
            self.compiled_image_encoder_embed = torch.jit.load(image_encoder_embed_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, "image_encoder_embed/image_encoder_embed.pt")).eval()
            self.compiled_image_encoder_embed = (
                mindietorch.compile(
                    model,
                    inputs=[
                        mindietorch.Input((batch_size, 3, 224, 224),dtype=mindietorch.dtype.FLOAT)
                        ],
                    allow_tensor_replace_int=True,
                    require_full_compilation=True,
                    truncate_long_and_double=True,
                    min_block_size=1,
                    soc_version="Ascend910B4",
                    precision_policy=_enums.PrecisionPolicy.FP16,
                    optimization_level=0
                    )
                )
            torch.jit.save(self.compiled_image_encoder_embed, image_encoder_embed_path)

        print(">>>>>>>>>>>>>>>image_encoder_embed2ts OK!")

        vae_encode_path = os.path.join(self.args.output_dir, "vae/vae_encode.ts")
        if os.path.exists(vae_encode_path):
            self.compiled_vae_encode = torch.jit.load(vae_encode_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, "vae/vae_encode.pt")).eval()
            self.compiled_vae_encode = (
                mindietorch.compile(
                    model,
                    inputs=[
                        mindietorch.Input((batch_size,3, height, width),dtype=mindietorch.dtype.FLOAT)
                        ],
                    allow_tensor_replace_int=True,
                    require_full_compilation=True,
                    truncate_long_and_double=True,
                    min_block_size=1,
                    soc_version="Ascend910B4",
                    precision_policy=_enums.PrecisionPolicy.FP16,
                    optimization_level=0
                    )
                )
            torch.jit.save(self.compiled_vae_encode, vae_encode_path)

        print(">>>>>>>>>>>>>>>vae_encode2ts OK!")

        model = torch.jit.load(os.path.join(self.args.output_dir, "vae/vae_decode.pt")).eval()
        if Dshape:
            vae_decode_path = os.path.join(self.args.output_dir, "vae/vae_decode.ts")
            if os.path.exists(vae_decode_path):
                self.compiled_vae_decode = torch.jit.load(vae_decode_path).eval()
            else:
                max_shape = (decode_chunk_size,channels_latents,height//vae_scale_factor,width//vae_scale_factor)
                min_shape = (res,channels_latents,height//vae_scale_factor,width//vae_scale_factor)
                inputs_vae = []
                inputs_vae.append([mindietorch.Input(max_shape,dtype=mindietorch.dtype.FLOAT)])
                if res !=0:
                    inputs_vae.append([mindietorch.Input(min_shape,dtype=mindietorch.dtype.FLOAT)])

                self.compiled_vae_decode = (
                    mindietorch.compile(
                        model,
                        inputs=inputs_vae,
                        allow_tensor_replace_int=True,
                        require_full_compilation=True,
                        truncate_long_and_double=True,
                        min_block_size=1,
                        soc_version="Ascend910B4",
                        precision_policy=_enums.PrecisionPolicy.FP16,
                        optimization_level=0
                        )
                    )
                torch.jit.save(self.compiled_vae_decode, vae_decode_path)
        else:
            vae_decode_path_8 = os.path.join(self.args.output_dir, "vae/vae_decode8.ts")
            vae_decode_path_1 = os.path.join(self.args.output_dir, "vae/vae_decode1.ts")
            if os.path.exists(vae_decode_path_8) & os.path.exists(vae_decode_path_1):
                self.compiled_vae_decode8 = torch.jit.load(vae_decode_path_8).eval()
                self.compiled_vae_decode1 = torch.jit.load(vae_decode_path_1).eval()
            else:
                max_shape = (decode_chunk_size,channels_latents,height//vae_scale_factor,width//vae_scale_factor)
                min_shape = (res,channels_latents,height//vae_scale_factor,width//vae_scale_factor)
                inputs_vae = []
                inputs_vae.append([mindietorch.Input(max_shape,dtype=mindietorch.dtype.FLOAT)])
                self.compiled_vae_decode8 = (
                    mindietorch.compile(
                        model,
                        inputs=inputs_vae,
                        allow_tensor_replace_int=True,
                        require_full_compilation=True,
                        truncate_long_and_double=True,
                        min_block_size=1,
                        soc_version="Ascend910B4",
                        precision_policy=_enums.PrecisionPolicy.FP16,
                        optimization_level=0
                        )
                    )
                torch.jit.save(self.compiled_vae_decode8, vae_decode_path_8)

                inputs_vae.clear()
                inputs_vae.append([mindietorch.Input(min_shape,dtype=mindietorch.dtype.FLOAT)])
                self.compiled_vae_decode1 = (
                    mindietorch.compile(
                        model,
                        inputs=inputs_vae,
                        allow_tensor_replace_int=True,
                        require_full_compilation=True,
                        truncate_long_and_double=True,
                        min_block_size=1,
                        soc_version="Ascend910B4",
                        precision_policy=_enums.PrecisionPolicy.FP16,
                        optimization_level=0
                        )
                    )
                torch.jit.save(self.compiled_vae_decode1, vae_decode_path_1)

        print(">>>>>>>>>>>>>>>vae_decode2ts OK!")

        unet_compile_path = os.path.join(self.args.output_dir, "unet/unet_bs1.ts")
        if os.path.exists(unet_compile_path):
            self.compiled_unet_model = torch.jit.load(unet_compile_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, "unet/unet_bs2.pt")).eval()

            self.compiled_unet_model = (
                mindietorch.compile(
                    model,
                    inputs=[
                        mindietorch.Input((batch_size*num_videos_per_prompt,num_frames,in_channels, height//vae_scale_factor,width//vae_scale_factor),dtype=mindietorch.dtype.FLOAT),
                        mindietorch.Input((1,),dtype=mindietorch.dtype.FLOAT),
                        mindietorch.Input((batch_size*num_videos_per_prompt,seq_len,vae_encode_out),dtype=mindietorch.dtype.FLOAT),
                        mindietorch.Input((batch_size*num_videos_per_prompt,3),dtype=mindietorch.dtype.FLOAT)
                        ],
                    allow_tensor_replace_int=True,
                    require_full_compilation=True,
                    truncate_long_and_double=True,
                    min_block_size=1,
                    soc_version="Ascend910B4",
                    precision_policy=_enums.PrecisionPolicy.FP16,
                    optimization_level=0
                    )
                )
            torch.jit.save(self.compiled_unet_model, unet_compile_path)

        runtime_info = RuntimeIOInfo(
            input_shapes=[
                (batch_size*num_videos_per_prompt,num_frames,in_channels, height//vae_scale_factor,width//vae_scale_factor),
                (1,),
                (batch_size*num_videos_per_prompt,seq_len,vae_encode_out),
                (batch_size*num_videos_per_prompt,3)
            ],
            input_dtypes=[np.float32, np.float32, np.float32, np.float32],
            output_shapes=[(batch_size*num_videos_per_prompt,num_frames,in_channels//2, height//vae_scale_factor,width//vae_scale_factor)],
            output_dtypes=[np.float32]
        )
        if hasattr(self, 'device_1'):
            self.unet_bg = BackgroundRuntime.clone(self.device_1, unet_compile_path, runtime_info)
            self.use_parallel_inferencing = True

        print(">>>>>>>>>>>>>>>unet2ts OK!")

        mindietorch.set_device(self.device_0)
        self.is_init = True

    @torch.no_grad()
    def ascendie_infer(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        self.calmse=torch.nn.MSELoss(reduction='mean')
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width).contiguous()
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs 创建时间嵌入向量
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, torch.float32)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        self._num_timesteps = len(timesteps)

        if self.use_parallel_inferencing and do_classifier_free_guidance:
            # Split embeddings
            negative_image_embeddings, image_embeddings = image_embeddings.chunk(2)
            added_time_ids_1, added_time_ids = added_time_ids.chunk(2)
            negative_image_latents, image_latents = image_latents.chunk(2)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            if not self.use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention [batch, num_frames, channels, height, width]
            negative_latent_model_input = torch.cat([latent_model_input, negative_image_latents], dim=2)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            # predict the noise residual
            if self.use_parallel_inferencing and do_classifier_free_guidance:
                self.unet_bg.infer_asyn([
                    latent_model_input.numpy(),
                    t[None].numpy(),
                    image_embeddings.numpy(),
                    added_time_ids.numpy(),
                ])

            noise_pred_uncond= self.compiled_unet_model(
                negative_latent_model_input.to(f'npu:{self.device_0}'),
                t[None].to(f'npu:{self.device_0}'),
                negative_image_embeddings.to(f'npu:{self.device_0}'),
                added_time_ids_1.to(f'npu:{self.device_0}'),
            ).to('cpu')

            # perform guidance
            if do_classifier_free_guidance:
                if self.use_parallel_inferencing:
                    noise_pred_cond = torch.from_numpy(self.unet_bg.wait_and_get_outputs()[0])
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size).to('cpu')
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        # run inference
        global image_embed_time
        start =time.time()

        image_embeddings = self.compiled_image_encoder_embed(image.to(device=f'npu:{self.device_0}', dtype=dtype)).to('cpu')

        image_embed_time +=time.time()-start

        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image_latents = self.compiled_vae_encode(image.to(f'npu:{self.device_0}')).to('cpu')

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            if Dshape:
                frame = self.compiled_vae_decode(latents[i : i + decode_chunk_size].to(f'npu:{self.device_0}')).to('cpu')
            else:
                if num_frames_in == decode_chunk_size:
                    frame = self.compiled_vae_decode8(latents[i : i + decode_chunk_size].to(f'npu:{self.device_0}')).to('cpu')
                else:
                    frame = self.compiled_vae_decode1(latents[i : i + decode_chunk_size].to(f'npu:{self.device_0}')).to('cpu')
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames


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
        default="./stable-video-diffusion-img2vid-xt",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--img_file",
        type=str,
        default="./rocket.png",
        help="A png file of prompts for generating vedio.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Path to save model pt.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help="FPS",
    )
    parser.add_argument(
        "--device",
        type=check_device_range_valid,
        default=[0, 1],
        help="NPU device id. Give 2 ids to enable parallel inferencing.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "-vp",
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="num_videos_per_prompt."
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=8,
        help="decode_chunk_size."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="num_inference_steps."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    decode_chunk_size=args.decode_chunk_size
    num_inference_steps=args.num_inference_steps

    pipe = AIEStableVideoDiffusionPipeline.from_pretrained(args.model).to("cpu")
    pipe.parser_args(args)

    pipe.compile_aie_model()

    # 加载img及预处理
    image = load_image(args.img_file)
    image = image.resize((heightS, widthS))

    print('warming up ~~~~~')
    stream = mindietorch.npu.Stream("npu:" + str(args.device[0]))
    with mindietorch.npu.stream(stream):
        frames = pipe.ascendie_infer(
            image,
            decode_chunk_size=decode_chunk_size, 
            height= heightS,
            width = widthS,
            num_inference_steps=num_inference_steps,
            num_frames = num_framesS
            ).frames[0]

    use_time = 0
    with mindietorch.npu.stream(stream):
        start_time = time.time()
        frames = pipe.ascendie_infer(
            image,
            decode_chunk_size=decode_chunk_size,
            height= heightS,
            width = widthS,
            num_inference_steps=num_inference_steps,
            num_frames = num_framesS
            ).frames[0]
        stream.synchronize()
    use_time += time.time() - start_time

    print("Stable video diffusion use time:{}. Save dir is {}".format(use_time/1,save_dir))
    import datetime
    now=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    export_to_video(frames, r"{}/rocket_910B4_{}.mp4".format(save_dir,now), fps=args.fps)

    if hasattr(pipe, 'device_1'):
        if (pipe.unet_bg):
            pipe.unet_bg.stop()

    mindietorch.finalize()


if __name__ == "__main__":
    main()
