import argparse
import csv
import json
import os
import time
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import torch
import mindietorch

import pdb
import pickle
from mindietorch import _enums
from diffusers import StableVideoDiffusionPipeline
import diffusers.models.transformer_temporal
from diffusers.utils import load_image, export_to_video
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing,_compute_padding,_filter2d,_gaussian,_gaussian_blur2d,_append_dims,inspect,tensor2vid,StableVideoDiffusionPipelineOutput
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler, SASolverScheduler


from background_runtime import BackgroundRuntime, RuntimeIOInfo

image_embed_time=0
accuracy_compare=True#False
pb_out=True
heightS = 576 #192#512
widthS = 1024 #192#512#1024
num_framesS=25
Dshape = False

print("height:{},width:{},num_frames:{},accuracy:{},pb_out:{},vae_decode dynamic shape:{}".format(heightS,widthS,num_framesS,accuracy_compare,pb_out,Dshape))

class AIEStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
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

        in_channels = self.unet.config.in_channels
        # sample_size = self.unet.config.sample_size
        batch_size = self.args.batch_size
        num_videos_per_prompt = 1
        # height = 192 #  or sample_size * vae_scale_factor
        # width = 192 # or sample_size * vae_scale_factor
        height = heightS
        width = widthS
        num_frames = num_framesS if num_framesS is not None else self.unet.config.num_frames
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        seq_len = 1
        vae_encode_out=1024
        decode_chunk_size=self.args.decode_chunk_size
        num_inference_steps=self.args.num_inference_steps
        # res=num_inference_steps%decode_chunk_size
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
            if pb_out:
                val=os.system('cp model.pb model_clip.pb')
                print("cp model.pb model_clip.pb:{}".format(val))

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
            if pb_out:
                val=os.system('cp model.pb model_vae_encode.pb')
                print("cp model.pb model_vae_encode.pb:{}".format(val))

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

                # inputs_vae.append(mindietorch.Input(min_shape = min_shape, max_shape= max_shape,dtype=mindietorch.dtype.FLOAT))
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
                if pb_out:
                    val=os.system('cp model.pb model_vae_decode.pb')
                    print("cp model.pb model_vae_decode.pb:{}".format(val))
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
                if pb_out:
                    val=os.system('cp model.pb model_vae_decode8.pb')
                    print("cp model.pb model_vae_decode8.pb:{}".format(val))
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
                if pb_out:
                    val=os.system('cp model.pb model_vae_decode1.pb')
                    print("cp model.pb model_vae_decode1.pb:{}".format(val))

        print(">>>>>>>>>>>>>>>vae_decode2ts OK!")


        unet_compile_path = os.path.join(self.args.output_dir, "unet/unet_bs1.ts")
        if os.path.exists(unet_compile_path):
            self.compiled_unet_model = torch.jit.load(unet_compile_path).eval()
        else:
            model = torch.jit.load(os.path.join(self.args.output_dir, "unet/unet_bs2.pt")).eval()###################
            
            # mindietorch.compile(
            # self.compiled_unet_model = (
            #     mindietorch.export_engine(
            #         model,
            #         "forward",
            #         inputs=[
            #             mindietorch.Input((batch_size*num_videos_per_prompt,num_frames,in_channels, height//vae_scale_factor,width//vae_scale_factor),dtype=mindietorch.dtype.FLOAT),
            #             mindietorch.Input((1,),dtype=mindietorch.dtype.FLOAT),
            #             mindietorch.Input((batch_size*num_videos_per_prompt,seq_len,vae_encode_out),dtype=mindietorch.dtype.FLOAT),
            #             mindietorch.Input((batch_size*num_videos_per_prompt,3),dtype=mindietorch.dtype.FLOAT)
            #             ],                                                
            #         allow_tensor_replace_int=True,
            #         require_full_compilation=True,
            #         truncate_long_and_double=True,
            #         min_block_size=1,
            #         soc_version="Ascend910B4",
            #         precision_policy=_enums.PrecisionPolicy.FP16,
            #         optimization_level=0
            #         )
            #     )
            # with open("./engine.om", 'wb') as file:
            #     file.write(self.compiled_unet_model)
            
            self.compiled_unet_model = (
                mindietorch.compile(
                    model,
                    inputs=[
                        mindietorch.Input((batch_size*num_videos_per_prompt,num_frames,in_channels, height//vae_scale_factor,width//vae_scale_factor),dtype=mindietorch.dtype.FLOAT),
                        mindietorch.Input((1,),dtype=mindietorch.dtype.FLOAT), #INT64
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
            if pb_out:
                val=os.system('cp model.pb model_unet.pb')
                print("cp model.pb model_unet.pb:{}".format(val))
            
        runtime_info = RuntimeIOInfo(
            input_shapes=[
                (batch_size*num_videos_per_prompt,num_frames,in_channels, height//vae_scale_factor,width//vae_scale_factor),
                (1,),
                (batch_size*num_videos_per_prompt,seq_len,vae_encode_out),
                (batch_size*num_videos_per_prompt,3)
            ],
            input_dtypes=[np.float32, np.float32, np.float32, np.float32], # 1 25 8 72 128
            output_shapes=[(batch_size*num_videos_per_prompt,num_frames,in_channels//2, height//vae_scale_factor,width//vae_scale_factor)], # 1 25 4 72 128
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
        height = height or self.unet.config.sample_size * self.vae_scale_factor #576 or 96*(vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)=8)=768
        width = width or self.unet.config.sample_size * self.vae_scale_factor # 1024 or 96*768

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames # none else 25 frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames # ts->8

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(image, height, width) #sd ascendie没有实现输入数据类型的检查，height和width是8的整数倍

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
        do_classifier_free_guidance = max_guidance_scale > 1.0 # =true 负提示词生效，和正提示词在batch维度拼接在一起

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance) # 2 1 1024
        # image_embeddings = torch.from_numpy(np.load('image_embeddings.npy'))#######################

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1 # todo 细节需要理解一下

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width).contiguous()        # 1 3 576 1024
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype) # 1 3 576 1024
        image = image + noise_aug_strength * noise                                                     # 给图像加噪声

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance) # 2 4 72 128，把图像压缩到隐空间
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)                                # 2 25 4 72 128，扩充帧数这个维度
        # image_latents = torch.from_numpy(np.load('image_latents.npy'))#######################

        # 5. Get Added Time IDs 创建时间嵌入向量
        added_time_ids = self._get_add_time_ids(                                                                 # 2 3
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)
        # added_time_ids = torch.from_numpy(np.load('added_time_ids.npy'))#######################

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # print("self.scheduler = ", self.scheduler)
        timesteps = self.scheduler.timesteps
        # print("timesteps = ", timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(                                                                         # 1 25 4 72 128
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
        # latents = torch.from_numpy(np.load('latents.npy'))#######################

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, torch.float32) # latents.dtype
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        # with self.progress_bar(total=num_inference_steps) as progress_bar:#todo存在疑问
            # for i, t in enumerate(timesteps):

        # print("<<<<<<<<<<<<<<<<image_latents.shape"f"{image_latents.shape}>>>>>>>>>>>>>>>>")        # 2 25 4 72 128
        # print("<<<<<<<<<<<<<<<<image_embeddings.shape"f"{image_embeddings.shape}>>>>>>>>>>>>>>>>")  # 2 1 1024
        # print("<<<<<<<<<<<<<<<<added_time_ids.shape"f"{added_time_ids.shape}>>>>>>>>>>>>>>>>")      # 2 3


        if self.use_parallel_inferencing and do_classifier_free_guidance:
            # Split embeddings
            negative_image_embeddings, image_embeddings = image_embeddings.chunk(2) ####################[negative_image_embeddings, image_embeddings]
            added_time_ids_1, added_time_ids = added_time_ids.chunk(2) ####################
            negative_image_latents, image_latents = image_latents.chunk(2) ####################([negative_image_latents, image_latents])
        
        # print("self.progress_bar(timesteps) = ", self.progress_bar(timesteps))
        # print("self.progress_bar(timesteps) = ", len(self.progress_bar(timesteps)))
        for i, t in enumerate(self.progress_bar(timesteps)):
            # print("t = ", t)
            # print("t.dtype = ", t.dtype)
            # expand the latents if we are doing classifier free guidance
            if not self.use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents     # 1 25 4 72 128
            
            # print("self.scheduler ===== ", self.scheduler)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention [batch, num_frames, channels, height, width]
            negative_latent_model_input = torch.cat([latent_model_input, negative_image_latents], dim=2)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            # out = torch.cat([negative_latent_model_input, latent_model_input])
            # print("<<<<<<<<<<<<<<<<out"f"{out}>>>>>>>>>>>>>>>>")
            # pdb.set_trace()

            # print("<<<<<<<<<<<<<<<<latent_model_input.shape"f"{latent_model_input.shape}>>>>>>>>>>>>>>>>")     # 1 25 4 72 128
            # print("<<<<<<<<<<<<<<<<negative_latent_model_input.shape"f"{negative_latent_model_input.shape}>>>>>>>>>>>>>>>>") # 1 25 8 72 128
            # print("<<<<<<<<<<<<<<<<image_embeddings.shape"f"{image_embeddings.shape}>>>>>>>>>>>>>>>>")         # 1 1 1024
            # print("<<<<<<<<<<<<<<<<added_time_ids.shape"f"{added_time_ids.shape}>>>>>>>>>>>>>>>>")             # 1 3

            # print("<<<<<<<<<<<<<<<<image_embeddings.type"f"{image_embeddings.dtype}>>>>>>>>>>>>>>>>")
            # print("<<<<<<<<<<<<<<<<t.type"f"{t.dtype}>>>>>>>>>>>>>>>>")
            # print("<<<<<<<<<<<<<<<<image_latents.type"f"{image_latents.dtype}>>>>>>>>>>>>>>>>")
            # print("<<<<<<<<<<<<<<<<added_time_ids.type"f"{added_time_ids.dtype}>>>>>>>>>>>>>>>>")

            # predict the noise residual
            # if accuracy_compare:
            #     with open("./pickle/unet_data1024.pkl",'rb') as file:
            #         unet_data = pickle.load(file)
            #     latent_model_input_3=unet_data["latent_model_input"]
            #     t_3=unet_data["t"]
            #     image_embeddings_3=unet_data["image_embeddings"]
            #     added_time_ids_3=unet_data["added_time_ids"]
            #     noise_pred_3= self.compiled_unet_model(
            #         latent_model_input_3.to(f'npu:{self.device_0}'),
            #         t_3[None].to(f'npu:{self.device_0}'),
            #         image_embeddings_3.to(f'npu:{self.device_0}'),
            #         added_time_ids_3.to(f'npu:{self.device_0}'),
            #         ).to('cpu')# return_dict=False, default值是True返回类型存在差异
            #     print("unet_data mse_torch:",self.calmse(noise_pred_3, unet_data["noise_pred"]))
            #     print("cos:",torch.nn.functional.cosine_similarity(noise_pred_3, unet_data["noise_pred"]))


            if self.use_parallel_inferencing and do_classifier_free_guidance:
                self.unet_bg.infer_asyn([
                    latent_model_input.numpy(),
                    t[None].numpy(),
                    image_embeddings.numpy(),
                    added_time_ids.numpy(),
                ])

            # print("negative_latent_model_input.dtype = ", negative_latent_model_input.dtype)
            # print("t[None].dtype = ", t[None].dtype)
            # print("negative_image_embeddings.dtype = ", negative_image_embeddings.dtype)
            # print("added_time_ids_1.dtype = ", added_time_ids_1.dtype)

            # noise_pred_uncond= self.compiled_unet_model(
            #     negative_latent_model_input.to(f'npu:{self.device_0}'),
            #     t[None].to(torch.float).to(f'npu:{self.device_0}'),
            #     negative_image_embeddings.to(f'npu:{self.device_0}'),
            #     added_time_ids_1.to(f'npu:{self.device_0}'),
            # ).to('cpu')# return_dict=False, default值是True返回类型存在差异

            noise_pred_uncond= self.compiled_unet_model(
                negative_latent_model_input.to(f'npu:{self.device_0}'),
                t[None].to(f'npu:{self.device_0}'),
                negative_image_embeddings.to(f'npu:{self.device_0}'),
                added_time_ids_1.to(f'npu:{self.device_0}'),
            ).to('cpu')# return_dict=False, default值是True返回类型存在差异

            # if self.use_parallel_inferencing and do_classifier_free_guidance:
            #     self.unet_bg.infer_asyn([
            #         negative_latent_model_input.numpy(),
            #         t[None].numpy(),#t.to(torch.int64)[None].numpy(),
            #         negative_image_embeddings.numpy(),
            #         added_time_ids_1.numpy(),
            #     ])

            # noise_pred_cond= self.compiled_unet_model(
            #     latent_model_input.to(f'npu:{self.device_0}'),
            #     t.to(torch.int64)[None].to(f'npu:{self.device_0}'),
            #     image_embeddings.to(f'npu:{self.device_0}'),
            #     added_time_ids.to(f'npu:{self.device_0}'),
            # ).to('cpu')# return_dict=False, default值是True返回类型存在差异

            # perform guidance
            if do_classifier_free_guidance:
                if self.use_parallel_inferencing:
                    noise_pred_cond = torch.from_numpy(self.unet_bg.wait_and_get_outputs()[0])
                    # print("<<<<<<<<<<<<<<<<out2222222>>>>>>>>>>>>>>>>")
                    # noise_pred_cond = noise_pred_cond.to('cpu')
                    # pdb.set_trace()
                # else:
                #     noise_pred, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # if do_classifier_free_guidance:
            #     if self.use_parallel_inferencing:
            #         noise_pred = torch.from_numpy(self.unet_bg.wait_and_get_outputs()[0])
            #     else:
            #         noise_pred, noise_pred_cond = noise_pred.chunk(2)
            #     noise_pred = noise_pred + self.guidance_scale * (noise_pred_cond - noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            if t != 0:
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # print("cos:",torch.nn.functional.cosine_similarity(latents, unet_data["noise_pred"]))
            # pdb.set_trace()

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()

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
        dtype = next(self.image_encoder.parameters()).dtype# torch.type

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0 #shape[1,3,576.1024],device='cpu' 

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        # run inference #self.image_encoder.eval()是否需要添加
        global image_embed_time
        start =time.time()
        # pdb.set_trace()
        """
        if accuracy_compare:
            with open("./pkl/embed_data192.pkl",'rb') as file:
                embed_data = pickle.load(file)
            image=embed_data['image']
            image_embeddings = self.compiled_image_encoder_embed(image.to(device=f'npu:{self.device_0}', dtype=dtype)).to('cpu')# self.image_encoder(image).image_embeds
            print("embed_data mse_torch:",self.calmse(image_embeddings,embed_data['image_embeddings']))
            print("cos:",torch.nn.functional.cosine_similarity(image_embeddings,embed_data['image_embeddings']))
        """

        image_embeddings = self.compiled_image_encoder_embed(image.to(device=f'npu:{self.device_0}', dtype=dtype)).to('cpu')# self.image_encoder(image).image_embeds

        image_embed_time +=time.time()-start

        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:# batch_size会变成2倍
            negative_image_embeddings = torch.zeros_like(image_embeddings)#产生一个全零的图片控制图片中不应该出现的元素

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image,# : torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        # image = image.to(device=device)
        # image = image.to(device=f'npu:{self.device_0}')
        # pdb.set_trace()
        """
        if accuracy_compare:
            with open("./pkl/vae_encode_data192.pkl",'rb') as file:
                vae_encode = pickle.load(file)
            image=vae_encode["image"].contiguous()
            image_latents = self.compiled_vae_encode(image.to(f'npu:{self.device_0}')).to('cpu')#self.vae.encode(image).latent_dist.mode()
            print("vae_encode mse_torch:",self.calmse(image_latents,vae_encode["image_latents"]))
            print("cos:",torch.nn.functional.cosine_similarity(image_latents,vae_encode["image_latents"]))
        """

        image_latents = self.compiled_vae_encode(image.to(f'npu:{self.device_0}')).to('cpu')#self.vae.encode(image).latent_dist.mode()

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
            # pdb.set_trace()

            """
            if accuracy_compare: 
                if num_frames_in == decode_chunk_size:
                    with open("./pkl/vae_decode_data8_192.pkl",'rb') as file:
                        vae_decode = pickle.load(file)
                else:
                    with open("./pkl/vae_decode_data1_192.pkl",'rb') as file:
                        vae_decode = pickle.load(file)
                frame = self.compiled_vae_decode(vae_decode["latent"].to(f'npu:{self.device_0}')).to('cpu')
                print("vae_decodeB mse_torch:",self.calmse(frame,vae_decode["frame"]))
                print("cos:",torch.nn.functional.cosine_similarity(frame,vae_decode["frame"]))
            """
            if Dshape:
                frame = self.compiled_vae_decode(latents[i : i + decode_chunk_size].to(f'npu:{self.device_0}')).to('cpu')#latents是npu
            else:
                if num_frames_in == decode_chunk_size:
                    frame = self.compiled_vae_decode8(latents[i : i + decode_chunk_size].to(f'npu:{self.device_0}')).to('cpu')#latents是npu
                else:
                    frame = self.compiled_vae_decode1(latents[i : i + decode_chunk_size].to(f'npu:{self.device_0}')).to('cpu')#latents是npu 走的此分支
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

def parse_arguments(): #todo
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/home/linh1/diff/lin_ascendInfer/test21_lu/stable-video-diffusion-img2vid-xt",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--img_file",
        type=str,
        default="/home/zhouwenxue/SD_zwx/test/ModelZoo-PyTorch/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion/results_2.1_nocache_parallel/12_0.png",  # ./rocket.png"、./women.png、elephant.png、
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
        default="./model_pt",
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
        default=[2, 3],
        help="NPU device id. Give 2 ids to enable parallel inferencing.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    # parser.add_argument(
    #     '--export_pt', 
    #     action="store_true",
    #     help=''
    # )
    # parser.add_argument(
    #     '--export_ts', 
    #     action="store_true",
    #     help=''
    # )
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
        default=10, #10、25
        help="num_inference_steps."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    # args.export_ts=True
    # if args.export_pt:
    #     args.export_ts = True
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if args.export_pt:
    #     export_to_pt(args)
    # if args.export_ts and pb_out:
    #     val=os.system('touch model.pb')
    #     print("touch model.pb:{}".format(val))

    decode_chunk_size=args.decode_chunk_size
    num_inference_steps=args.num_inference_steps
    
    # pipe = AIEStableVideoDiffusionPipeline.from_pretrained(args.model).to("npu:0",dtype=torch.FP16)
    pipe = AIEStableVideoDiffusionPipeline.from_pretrained(args.model).to("cpu")

    # pipe.enable_model_cpu_offload()
    pipe.parser_args(args)

    # if args.scheduler == "SA-Solver":
    # pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config)

    pipe.compile_aie_model()
    # mindietorch.set_device(args.device)############################
    ## todo
    # pipe.enable_model_cpu_offload()

    # generator = torch.Generator().manual_seed(2023)#######################

    # ## 加载img及预处理
    image = load_image(args.img_file)
    image = image.resize((heightS, widthS))

    print('warming up ~~~~~')
    # for i in range(2):
    #     print(i)
    stream = mindietorch.npu.Stream("npu:" + str(args.device[0]))
    with mindietorch.npu.stream(stream):#上下文管理器将内部的操作发送到指定的stream上执行，进行并行操作
        frames = pipe.ascendie_infer(
            image,
            decode_chunk_size=decode_chunk_size, 
            # generator=generator,
            height= heightS,
            width = widthS,
            num_inference_steps=num_inference_steps,
            num_frames = num_framesS
            ).frames[0]

    use_time = 0
    with mindietorch.npu.stream(stream):#上下文管理器将内部的操作发送到指定的stream上执行，进行并行操作###################
        start_time = time.time()
        frames = pipe.ascendie_infer(
            image,
            decode_chunk_size=decode_chunk_size, 
            # generator=generator,
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
    # export_to_video(frames, r"{}/rocket_910B2_{}.mp4".format(save_dir,int(time.mktime(time.localtime()))), fps=args.fps)
    export_to_video(frames, r"{}/rocket_910B2_{}.mp4".format(save_dir,now), fps=args.fps)

    if hasattr(pipe, 'device_1'):
        if (pipe.unet_bg):
            pipe.unet_bg.stop()

    mindietorch.finalize()

# for i in range(count):
#     with mindietorch.npu.stream(stream):
#         model_out_npu = aie_model(x1.to(f'npu:{device}'),
#                     x2.to(f'npu:{device}'),
#                     x3.to(f'npu:{device}'))
#         stream.synchronize()
#     model_out_cpu = model_out_npu.to("cpu")
if __name__ == "__main__":
    main()
