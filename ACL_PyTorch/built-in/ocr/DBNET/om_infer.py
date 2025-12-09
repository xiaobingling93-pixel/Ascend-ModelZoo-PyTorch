# Copyright © 2021 - 2025. Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ais_bench.infer.summary import summary
from ais_bench.infer.interface import InferSession, MemorySummary


def get_np_files(npy_dir):
    npy_paths = []
    for filename in os.listdir(npy_dir):
        if filename.endswith(".npy"):
            full_path = os.path.join(npy_dir, filename)
            npy_paths.append(full_path)
    return npy_paths


class NumpyImageDataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        self.npy_paths = get_np_files(npy_dir)
        self.transform = transform

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        image = np.load(self.npy_paths[idx])
        return torch.tensor(image), self.npy_paths[idx]


def pad_collate(batch, bs_set):
    data = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    max_h = max([img.shape[2] for img in data])
    max_w = max([img.shape[3] for img in data])
    padded_datas = []

    for img in data:
        _, h, w = img.shape[1], img.shape[2], img.shape[3]
        padding_top_bottom = max_h - h
        padding_left_right = max_w - w
        padded_img = torch.nn.functional.pad(
            img,
            (0, padding_left_right, 0, padding_top_bottom),
            "constant",
            0
        )
        padded_datas.append(padded_img.squeeze(0))

    batch_size = len(padded_datas)
    if batch_size == bs_set:
        return torch.stack(padded_datas), paths
    num_to_add = bs_set - batch_size
    zero_tensor = torch.zeros(padded_datas[0].shape)
    additional_samples = [zero_tensor.clone() for _ in range(num_to_add)]
    final_batch = padded_datas + additional_samples
    return torch.stack(final_batch), paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,
                        help='Device you want to use. Such as 0.')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--preped_path', type=str, default="./prep_dataset/", 
                        help='The path of dataset prepared')
    parser.add_argument('--output_path', type=str, default='./outputs/',
                        help='The path of output binary files')
    
    args = parser.parse_args()
    preped_path = Path(args.preped_path).resolve(strict=False)
    output_path = Path(args.output_path).resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    dataset = NumpyImageDataset(preped_path)
    custom_collate = partial(pad_collate, bs_set=args.batchsize)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, collate_fn=custom_collate)

    isAcurency = True

    session = InferSession(device_id=args.device, model_path=f"db_bs{args.batchsize}.om")
    files = os.listdir(preped_path)

    session.reset_summaryinfo()
    MemorySummary.reset()
    summary.add_args(sys.argv)
    other_time = 0
    end = time.perf_counter()

    for images, image_paths in dataloader:
        outputs = session.infer(feeds=[images], mode="dymdims")
        start_time = time.perf_counter()
        if isAcurency:
            for i, image in enumerate(image_paths):
                filename = f"{image.split('/')[-1].split('.')[0]}_0.bin"
                name = output_path / filename
                outputs[0][i].tofile(name)
        end_time = time.perf_counter()
        other_time += end_time - start_time

    s = session.summary()
    summary.npu_compute_time_list = [end_time - start_time for start_time, end_time in s.exec_time_list]
    summary.h2d_latency_list = MemorySummary.get_h2d_time_list()
    summary.d2h_latency_list = MemorySummary.get_d2h_time_list()
    summary.report(
        batchsize=args.batchsize,
        output_prefix=None,
        display_all_summary=True,
        multi_threads=False
        )

