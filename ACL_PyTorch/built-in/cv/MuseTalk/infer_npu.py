# This file was copied and refactored from project https://github.com/TMElyralab/MuseTalk
# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import argparse
import os
import glob
import pickle
import copy
import shutil
import time
import numpy as np
import cv2
import torch
import torch_npu
import torchair as tng

from tqdm import tqdm
from omegaconf import OmegaConf
from torchair.configs.compiler_config import CompilerConfig
from torch_npu.contrib import transfer_to_npu

from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
from scripts.rewrite_models import rewirte_Unet, rewrite_VAE

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("npu" if torch.npu.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)


def compile_model():
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    models = [{"model": audio_processor.model, "target": "encoder", "dynamic": False},
              {"model": vae.vae, "target": "encoder", "dynamic": False},
              {"model": unet, "target": "model", "dynamic": True},
              {"model": vae.vae, "target": "decoder", "dynamic": True}]
    for model in models:
        compiled_model = torch.compile(getattr(model["model"], model["target"]),
                                       dynamic=model["dynamic"],
                                       fullgraph=True,
                                       backend=npu_backend)
        setattr(model["model"], model["target"], compiled_model)
        tng.use_internal_format_weight(getattr(model["model"], model["target"]))


def extract_from_video_audio(video_path, audio_path, input_basename, args):
    ############################################## extract frames from video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        if os.path.isfile(video_path) and os.path.isdir(save_dir_full):
            os.system(cmd)
        else:
            raise ValueError(f"{video_path} should be a video file, {save_dir_full} should be a dir")
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    elif get_file_type(video_path) == "image":
        input_img_list = [video_path, ]
        fps = args.fps
    elif os.path.isdir(video_path):  # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    else:
        raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
    
    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
    
    return input_img_list, whisper_chunks, fps


def preprocess_input_image(input_img_list, bbox_shift, crop_coord_save_path, args):
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
            
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    
    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

    return frame_list_cycle, coord_list_cycle, input_latent_list_cycle


def preprocess(inference_config, task_id, args):
    video_path = inference_config[task_id]["video_path"]
    audio_path = inference_config[task_id]["audio_path"]
    bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_img_save_path, input_basename + ".pkl") # only related to video input
    os.makedirs(result_img_save_path, exist_ok=True)
    
    if args.output_vid_name is None:
        output_vid_name = os.path.join(args.result_dir, output_basename + ".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
    
    input_img_list, whisper_chunks, fps = extract_from_video_audio(video_path, audio_path, input_basename, args)
    
    frame_list_cycle, coord_list_cycle, input_latent_list_cycle = \
        preprocess_input_image(input_img_list, bbox_shift, crop_coord_save_path, args)

    pd = {
        "frame_list_cycle": frame_list_cycle,
        "coord_list_cycle": coord_list_cycle,
        "input_latent_list_cycle": input_latent_list_cycle,
        "whisper_chunks": whisper_chunks,
        "output_vid_name": output_vid_name,
        "result_img_save_path": result_img_save_path,
        "audio_path": audio_path,
        "fps": fps
    }
    return pd


def postprocess(res_frame_list, postparm):
    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    frame_list_cycle = postparm.get("frame_list_cycle")
    coord_list_cycle = postparm.get("coord_list_cycle")
    output_vid_name = postparm.get("output_vid_name")
    result_img_save_path = postparm.get("result_img_save_path")
    audio_path = postparm.get("audio_path")
    fps = postparm.get("fps")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i % (len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception as ep:
            print(f"Exception: {ep}, error in resize the {i}'th res_frame, bbox: {bbox}")
            continue
        
        combine_frame = get_image(ori_frame, res_frame, bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)
    
    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)
    
    os.remove("temp.mp4")
    shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")


def infer(input_latent_list_cycle, whisper_chunks):
    ############################################## inference batch by batch ##############################################
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
    res_frame_list = []
    for _, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                        dtype=unet.model.dtype) # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch, return_dict=False)[0]
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    return res_frame_list


def main(args):
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)
    for task_id in inference_config:
        pd = preprocess(inference_config, task_id, args)

        res_frame_list = infer(pd.get("input_latent_list_cycle"), pd.get("whisper_chunks"))

        postprocess(res_frame_list, pd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results', help="path to output")

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument('--warmup', type=int, default=2, help="Warm up times")
    parser.add_argument('--loop', type=int, default=3, help="loop times")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord",
                        action="store_true",
                        help='use saved coordinate to save time')
    parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )

    rewirte_Unet(unet.model)
    rewrite_VAE(vae.vae)

    args = parser.parse_args()
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    compile_model()
    
    # warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            main(args)

    start = time.time()
    for _ in range(args.loop):
        with torch.no_grad():
            main(args)
    print(f'E2E time = {(time.time() - start) / args.loop *1000}ms')