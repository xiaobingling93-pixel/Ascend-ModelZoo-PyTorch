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
import os
import sys
import time
import math
import numpy as np
import torch
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from ldm.modules.diffusionmodules.util import timestep_embedding
from diffusers import StableDiffusionPipeline
from config import NpuConfig
from ais_bench.infer.interface import InferSession
from background_session import BackgroundInferSession
from modules import shared

def env_om(cur_dir_path):
    os.system(os.path.join(cur_dir_path, "setup.sh"))

def replace_onnx():
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    env_om(cur_dir_path)
    
    device_0, device_1 = 2, None
    if NpuConfig.unet_session_bg:
        NpuConfig.unet_session_bg.stop()
        NpuConfig.unet_session_bg = False
    print("You can generate image now!")

    def unet_onnx(self, x, timesteps=None, context=None, y=None, **kwargs):
        if x.shape[-1] != 64: # 64 = 512 // 8
            return x

        checkpoint = shared.opts.data['sd_model_checkpoint']
        if NpuConfig.use_parallel_inferencing:
            device_1 = 3
            unet_path = os.path.join(cur_dir_path, "models/SD2.1/models_bs1_parallel/unet/unet.om")
        else:
            unet_path = os.path.join(cur_dir_path, "models/SD2.1/models_bs1/unet/unet.om")
        
        if not NpuConfig.unet_session:
            NpuConfig.unet_session = InferSession(device_0, unet_path)
            if NpuConfig.use_parallel_inferencing:
                NpuConfig.unet_session_bg = BackgroundInferSession.clone(NpuConfig.unet_session, device_1)
        if NpuConfig.use_parallel_inferencing:
            context, context_2 = context.chunk(2)
            x, x_2 = x.chunk(2)
            NpuConfig.unet_session_bg.infer_asyn(
                [
                    x_2.cpu().numpy(),
                    timesteps[0][None].cpu().numpy().astype(np.int32),
                    context_2.cpu().numpy()
                ]
            )
        x = x.cpu().numpy()
        t = timesteps[0][None].cpu().numpy().astype(np.int32)
        context = context.cpu().numpy()
        noise_pred = torch.from_numpy(
            NpuConfig.unet_session.infer(
                [
                    x,
                    t,
                    context
                ]  
            )[0]
        )
        if NpuConfig.use_parallel_inferencing:
            noise_pred_text = torch.from_numpy(
                NpuConfig.unet_session_bg.wait_and_get_outputs()[0]
            )
            noise_pred = torch.cat([noise_pred, noise_pred_text])
        
        return noise_pred
    UNetModel.forward = unet_onnx

    def clip_onnx(self, text):
        clip_path = os.path.join(cur_dir_path, "models/SD2.1/models_bs1/clip/clip.om")
        if not NpuConfig.clip_session:
            NpuConfig.clip_session = InferSession(device_0, clip_path)
        x = torch.from_numpy(NpuConfig.clip_session.infer([text.numpy()])[0])
        return x
    FrozenOpenCLIPEmbedder.encode_with_transformer = clip_onnx

    def vae_onnx(self, z):
        vae_path = os.path.join(cur_dir_path, "models/SD2.1/models_bs1/vae/vae.om")
        z = self.post_quant_conv(z)
        if not NpuConfig.vae_session:
            NpuConfig.vae_session = InferSession(device_0, vae_path)
        dec = torch.from_numpy(NpuConfig.vae_session.infer([z.numpy()])[0])
        return dec
    AutoencoderKL.decode = vae_onnx