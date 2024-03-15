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
import numpy as np
from numpy.linalg import norm
import time
import argparse
from PIL import Image
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video


class AliDataset(Dataset):
    def __init__(self, root, anno_file):
        super().__init__()
        self.root = root
        self.anno_file = anno_file

        self.path_list = []
        with open(self.anno_file, 'r') as fp:
            for line in fp:
                self.path_list.append(line.rstrip())

    def __getitem__(self, index):
        img_name = self.path_list[index]
        imgpath = os.path.join(self.root, img_name)
        return imgpath

    def __len__(self):
        return len(self.path_list)


def make_test_data_sampler(dataset, distributed, rank):
    if distributed:
        return DistributedSampler(
                   dataset,
                   num_replicas=dist.get_world_size(),
                   rank=rank,
                   shuffle=False
               )
    else:
        return torch.utils.data.sampler.SequentialSampler(dataset)


def numpy_cosine_similarity_distance(a, b):
    if a.dtype == 'uint8' or b.dtype == 'uint8':
        similarity = np.dot(a.astype('float'), b.astype('float')) / (norm(a) * norm(b))
    else:
        similarity = np.dot(a, b) / (norm(a) * norm(b))

    distance = 1.0 - similarity.mean()

    return distance

def main(args):
    seed = args.seed
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("npu")
    batch_size = args.global_batch_size
    rank = 0
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        dist.init_process_group("hccl")
        rank = dist.get_rank()
        device = rank % torch.npu.device_count()
        torch.npu.set_device(device)
        generator = torch.Generator(device="cpu").manual_seed(seed + rank)
        print(f"Starting rank={rank}, seed={seed + rank}, world_size={dist.get_world_size()}.")
        batch_size = int(args.global_batch_size // dist.get_world_size())

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload(device=device)
    pipe.enable_npu_svd_attention()

    test_dataset = AliDataset(args.test_data_dir, args.test_file)
    sampler = make_test_data_sampler(test_dataset, distributed, rank)
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    for image_path in loader:
        for img in image_path:
            image = load_image(img)
            image = image.resize(args.image_size)
            start_time = time.time()
            frames = pipe(image, decode_chunk_size=8, generator=generator, num_frames=args.num_frames, num_inference_steps=25,
                          output_type="pil").frames[0]
            step_time = time.time()
            if not distributed or dist.get_rank() == 0:
                print(f'infer step time: {step_time - start_time}')
            if not os.path.exists(os.path.join(args.output_dir, img.split('/')[-1].split('.')[0])):
                os.makedirs(os.path.join(args.output_dir, img.split('/')[-1].split('.')[0]))
            for i in range(len(frames)):
                frames[i].save(os.path.join(args.output_dir, img.split('/')[-1].split('.')[0], f"frames_{i}.png"))
            if args.export_video:
                export_to_video(frames, os.path.join(args.output_dir, img.split('/')[-1].split('.')[0]+".mp4"), fps=7)

    if args.eval_metrics and os.path.isdir(args.benchmark_dir):
        if not distributed or dist.get_rank() == 0:
            cos_distances = []
            output_paths = sorted(os.listdir(args.output_dir))
            for subdir in output_paths:
                for i in range(args.num_frames):
                    img_benchmark = Image.open(os.path.join(args.benchmark_dir, subdir, f"frames_{i}.png"))
                    img_output = Image.open(os.path.join(args.output_dir, subdir, f"frames_{i}.png"))
                    img_benchmark = np.array(img_benchmark)
                    img_output = np.array(img_output)
                    cos_distance = numpy_cosine_similarity_distance(img_output.flatten(), img_benchmark.flatten())
                    cos_distances.append(cos_distance)

            print(f"mean cos dis: {sum(cos_distances) / len(cos_distances)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data-dir", type=str, default="", help='the path to testset')
    parser.add_argument("--test-file", type=str, default="", help='the path to testdata file')
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=tuple, default=(1024, 576))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt",
                        help='the path to a SVD checkpoint')
    parser.add_argument("--output-dir", type=str, default="", help='the path to save outputs')
    parser.add_argument("--eval-metrics", type=bool, default=False, help='whether or not to eval metrics')
    parser.add_argument("--benchmark-dir", type=str, default="")
    parser.add_argument("--export-video", type=bool, default=False)
    parser.add_argument("--num-frames", type=int, default=25)
    args = parser.parse_args()
    main(args)
