# Copyright 2024 Huawei Technologies Co., Ltd
import torch_npu
from torch_npu.contrib import transfer_to_npu
import functools
import itertools
import logging
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from argparse import ArgumentParser
import multiprocessing as mp

import numpy as np
import torch

import torchvision

import transformers
from decord import VideoReader, cpu

from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IMAGE_TOKEN='<image>'
from tasks.eval.videoqabench import (
    VideoQABenchDataset,
    load_results,
    save_results,
)
RESOLUTION = 672 # 
VIDEOQA_DATASETS=["MSVD_QA","MSRVTT_QA", "ActivityNet","TGIF_QA"]
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        default='llava-hf/llava-1.5-7b-hf'
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        default='"./test_results/test_llava_mvbench"'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        default=4,
    )
    parser.add_argument(
        "--use_lora",
        action='store_true'
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        required=False,
        default="gpt-3.5-turbo-0125",
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        required=False,
        default=1
    )
    parser.add_argument(
        "--conv_mode", 
        type=str,
        required=False,
        default='eval_videoqabench',
    )
    parser.add_argument(
        "--test_datasets", 
        type=str,
        required=False,
        default='MSVD_QA',
    )
    parser.add_argument(
        "--example_path",
        type=str,
        required=True,
        default='/path_to_video_file',
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        required=True,
        default=1,
    )
    args = parser.parse_args()
    return args

def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora, lora_alpha, weight_dir, test_ratio):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora, lora_alpha=lora_alpha, weight_dir=weight_dir)
    logger.info('done loading llava')
    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()
    return model, processor


def single_test(model, processor, vid_path, num_frames=4, conv_mode="plain", eval_mode=1):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(video_path, num_segments=8, return_msg=False, num_frames=4, resolution=336):
        transforms = torchvision.transforms.Resize(size=resolution)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(transforms(img))
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return images_group, msg
        else:
            return images_group

    if num_frames != 0:
        vid, msg = load_video(vid_path, num_segments=num_frames, return_msg=True, resolution=RESOLUTION)
    else:
        vid, msg = None, 'num_frames is 0, not inputing image'
    img_list = vid
    conv = conv_templates[conv_mode].copy()
    if eval_mode == 1:
        query_question = input("question input:")
        conv.user_query(query_question, is_mm=True)
    else:
        conv.user_query("Describe the video in details.", is_mm=True)
    llm_response, conv = pllava_answer(conv=conv, model=model, processor=processor, do_sample=False, img_list=img_list, max_new_tokens=256, print_res=True)

def main():
    multiprocess=True
    mp.set_start_method('spawn',force=True)
    args = parse_args()
    save_path = args.save_path
    eval_model = args.eval_model
    logger.info(f'trying loading results from {save_path}')
    result_list = load_results(save_path)
    vid_path = args.example_path
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    model, processor = load_model_and_dataset(0,
                           world_size,
                           pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                           num_frames=args.num_frames,
                           use_lora=args.use_lora,
                           lora_alpha=args.lora_alpha,
                           weight_dir=args.weight_dir,
                           test_ratio=args.test_ratio,
                           )
    single_test(model, processor, vid_path, num_frames=args.num_frames, conv_mode=args.conv_mode)
    logger.info('single test done...')

if __name__ == "__main__":
    main()