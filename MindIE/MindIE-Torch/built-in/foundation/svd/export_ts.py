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

import os
import argparse
from argparse import Namespace

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from diffusers import StableVideoDiffusionPipeline


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./model_pt",
        help="Path of directory to save pt models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/home/linh1/diff/lin_ascendInfer/test21_lu/stable-video-diffusion-img2vid-xt",
        help="Path or name of the pre-trained model.",
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
    # parser.add_argument(
    #     "-guid",
    #     "--guidance_scale",
    #     type=float,
    #     default=7.5,
    #     help="guidance_scale"
    # )
    # parser.add_argument(
    #     "--use_cache", 
    #     action="store_true",
    #     help="Use cache during inference."
    # )
    # parser.add_argument(
    #     "-p",
    #     "--parallel",
    #     action="store_true",
    #     help="Export the unet of bs=1 for parallel inferencing.",
    # )

    return parser.parse_args()


class Embedexport(torch.nn.Module):
    def __init__(self, embed_model):
        super().__init__()
        self.embed_model = embed_model

    def forward(self, image):
        # dtype = next(self.embed_model.parameters()).dtype
        # image = image.to(device=device, dtype=dtype)
        return self.embed_model(image).image_embeds


def export_image_embeddings(svd_pipeline: StableVideoDiffusionPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image embedding...")
    embed_path = os.path.join(save_dir, "image_encoder_embed")
    if not os.path.exists(embed_path):
        os.makedirs(embed_path, mode=0o640)

    embed_pt_path = os.path.join(embed_path, "image_encoder_embed.pt")
    if os.path.exists(embed_pt_path):
        return

    embed_model = svd_pipeline.image_encoder#text_encoder

    dummy_input = torch.ones([batch_size, 3, 224, 224], dtype=torch.float32)#224是训练数据的size
    embed_export = Embedexport(embed_model)

    torch.jit.trace(embed_export, dummy_input).save(embed_pt_path)


class Unetexport(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(self, sample, timestep, encoder_hidden_states, added_time_ids):
        # return_dict=False返回元组更快
        return self.unet_model(sample, timestep, encoder_hidden_states, added_time_ids, False)[0]


def export_unet(svd_pipeline: StableVideoDiffusionPipeline, save_dir: str, batch_size: int, num_videos_per_prompt: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    unet_pt_path = os.path.join(unet_path, f"unet_bs{batch_size}.pt")
    if os.path.exists(unet_pt_path):
        return

    unet_model = svd_pipeline.unet
    # sample_size = unet_model.config.sample_size

    num_frames = 25 # if 25 is not None else unet_model.config.num_frames
    vae_scale_factor = 2 ** (len(svd_pipeline.vae.config.block_out_channels) - 1)
    # height = 192 or svd_pipeline.unet.config.sample_size * vae_scale_factor
    # width = 192 or svd_pipeline.unet.config.sample_size * vae_scale_factor
    height = 192#512#heightS//2
    width = 192#512#widthS//2
    # num_videos_per_prompt = 1 这个是啥？ 动态静态  seq_len 是 vae.encode 输出维度
    seq_len = 1
    vae_encode_out=1024
    in_channels = unet_model.config.in_channels  # noise latents   image latents 

    # do_classifier_free_guidance = max_guidance_scale > 1.0
    do_classifier_free_guidance = True

    dummy_input = (
        torch.ones([batch_size*num_videos_per_prompt, num_frames, in_channels, height//vae_scale_factor, width//vae_scale_factor], dtype=torch.float32),
        torch.ones([1], dtype=torch.float32),
        torch.ones([batch_size*num_videos_per_prompt, seq_len, vae_encode_out], dtype=torch.float32),
        torch.ones([batch_size*num_videos_per_prompt, 3], dtype=torch.float32),
    )

    unet = Unetexport(unet_model)
    unet.eval()  #切换为 eval 模式，不计算梯度

    torch.jit.trace(unet, dummy_input).save(unet_pt_path)


class VaeExportDecode(torch.nn.Module):
    # num_frames: Dict[str, int]
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        #input [8,4,72,128] output[8,3,576,1024]
        # tt=self.vae_model.decode(latents,latents.shape[0]).sample
        # print("input shape:",latents.shape)
        # print("output shape:",tt.shape)
        # return tt
        num_frames={}
        num_frames["num_frames"]=latents.shape[0]
        return self.vae_model.decode(latents,**num_frames).sample


class VaeExportEncode(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, image:torch.Tensor):
        # image = image.to(device=device),input [1,3,576,1024]-output:[1,4,72,128]
        # tt=self.vae_model.encode(image).latent_dist.mode()
        # print("encode:",tt.shape)
        # print("device:",tt.device)#device:cpu
        # return tt
        return self.vae_model.encode(image).latent_dist.mode()


def export_vae(svd_pipeline: StableVideoDiffusionPipeline, save_dir: str, batch_size: int, decode_chunk_size: int) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)

    vae_pt_path = os.path.join(vae_path, "vae_encode.pt")
    vae_pt_path_2 = os.path.join(vae_path, "vae_decode.pt")
    if os.path.exists(vae_pt_path) & os.path.exists(vae_pt_path_2):
        return

    vae_model = svd_pipeline.vae
    unet_model = svd_pipeline.unet

    sample_size = unet_model.config.sample_size
    # in_channels = unet_model.config.out_channels

    channels_latents=unet_model.config.in_channels // 2
    vae_scale_factor = 2 ** (len(vae_model.config.block_out_channels) - 1)
    # height = 192 or sample_size * vae_scale_factor
    # width = 192 or sample_size * vae_scale_factor
    height = 192
    width = 192
    # height = sample_size * vae_scale_factor
    # width = sample_size * vae_scale_factor
    height_ld=height // vae_scale_factor
    width_ld=width //vae_scale_factor
    # decode_kwargs = {}
    # [1,25,4,72,128]->flatten->[25,4,72,128]->decode_chunk_size->[8,4,72,128]
    # decode_kwargs["num_frames"] = decode_chunk_size
    # for decode_chunk_size in decode_chunk_size_list:
    #     dummy_input = torch.ones([decode_chunk_size, channels_latents, height_ld, width_ld],dtype=torch.float32)
    #     # print(batch_size, in_channels, sample_size, sample_size)
    #     vae_export_decode = VaeExportDecode(vae_model)
    #     trace_model_decode=torch.jit.trace(vae_export_decode, dummy_input)
    #     trace_model_decode.save(os.path.join(vae_path, f"vae_decode_dcs{decode_chunk_size}.pt"))
    # min_shape=(1,channels_latents, height_ld, width_ld)
    # max_shape=(decode_chunk_size,channels_latents, height_ld, width_ld)
    # inputs = []
    # inputs.append(mindietorch.Input(min_shape = min_shape, max_shape= max_shape))
    dummy_input = torch.ones([1, channels_latents, height_ld, width_ld],dtype=torch.float32)
    vae_export_decode = VaeExportDecode(vae_model)
    trace_model_decode=torch.jit.trace(vae_export_decode, dummy_input)
    trace_model_decode.save(vae_pt_path_2)

    print("Exporting the image encoder...")

    dummy_input=torch.ones([1, 3, height, width])
    vae_export_encode = VaeExportEncode(vae_model)
    trace_model_encode=torch.jit.trace(vae_export_encode, dummy_input)
    trace_model_encode.save(vae_pt_path)


def export_to_pt(model_path: str, save_dir: str, batch_size: int, num_inference_steps: int, decode_chunk_size: int, num_videos_per_prompt: int) -> None:

    pipeline = StableVideoDiffusionPipeline.from_pretrained(model_path).to("cpu")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o640)
    # num_videos_per_prompt=info.num_videos_per_prompt
    # decode_chunk_size=info.decode_chunk_size
    # num_inference_steps=info.num_inference_steps
    # decode_chunk_size_list=[decode_chunk_size,num_inference_steps%decode_chunk_size]
    print(">>>>>>>>>>>>>>>embedding!")
    export_image_embeddings(pipeline, save_dir, batch_size)

    print(">>>>>>>>>>>>>>>VAE!")
    export_vae(pipeline, save_dir, batch_size, decode_chunk_size)

    # print(">>>>>>>>>>>>>>>UNET!")
    # # 单卡不带unetcache
    # export_unet(pipeline, save_dir, batch_size * 2, num_videos_per_prompt)
    # print(">>>>>>>>>>>>>>>UNET parallel!")
    # 双卡不带unetcache
    # export_unet(pipeline, save_dir, batch_size, num_videos_per_prompt)


def main():
    args = parse_arguments()
    export_to_pt(args.model, args.output_dir, args.batch_size, args.num_inference_steps, args.decode_chunk_size, args.num_videos_per_prompt)
    print("Done.")


if __name__ == "__main__":
    main()
