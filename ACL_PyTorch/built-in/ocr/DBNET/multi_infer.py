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

import sys
import time
import argparse
import multiprocessing
from functools import partial
from multiprocessing import Manager

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ais_bench.infer.interface import InferSession, MemorySummary
from ais_bench.infer.summary import summary, Summary


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
        return torch.stack(padded_datas)
    num_to_add = bs_set - batch_size
    zero_tensor = torch.zeros(padded_datas[0].shape)
    additional_samples = [zero_tensor.clone() for _ in range(num_to_add)]
    final_batch = padded_datas + additional_samples
    return torch.stack(final_batch)


def consume_data(device_id, data_queue, bs):
    cnt = 0
    print(f"Device {device_id} starts consuming data.")
    is_session_ready = False
    while True:
        if data_queue.empty():
            break
        images = data_queue.get(timeout=1)
        if not is_session_ready:
            session = InferSession(device_id=device_id, model_path=f"db_bs{bs}.om")
            is_session_ready = True
            session.reset_summaryinfo()
            memo_summary = MemorySummary()
            memo_summary.reset()
            summary.add_args(sys.argv)
        cnt += 1
        session.infer(feeds=[images], mode="dymdims")
        if cnt % 10 == 0:
            print(f'The {cnt}th inference of device {device_id} has been completed.')

    s = session.summary()
    summary.npu_compute_time_list = [end_time - start_time for start_time, end_time in s.exec_time_list]
    summary.h2d_latency_list = memo_summary.get_h2d_time_list()
    summary.d2h_latency_list = memo_summary.get_d2h_time_list()
    print(f'The inferences of device {device_id} has all completed.')
    summaries[device_id] = summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0,1', \
                        help='Devices you want to use. Such as "0,1,2,3" ')
    parser.add_argument('--preped_path', type=str, default="./prep_dataset/", 
                        help='The path of dataset prepared')
    parser.add_argument('--batchsize', type=int, default=1)
    args = parser.parse_args()


    manager = Manager()
    dataset = NumpyImageDataset(args.preped_path)
    custom_collate = partial(pad_collate, bs_set=args.batchsize)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, collate_fn=custom_collate)
    queue = multiprocessing.Queue()
    print('Loading dataset ...')
    data_nums = 0
    for batch in dataloader:
        queue.put(batch)
        data_nums += 1
    print('Dataset has been loaded completed, sum:', data_nums)

    # Change type  of device ids from strings into list of int. 
    devices = list(map(int, args.device.split(',')))
    num_processes = len(devices)

    temp_summaries = [Summary() for _ in range(num_processes)]
    summaries = manager.list(temp_summaries)

    is_session_ready = [False] * num_processes

    
    processes = []
    for i in devices:
        p = multiprocessing.Process(target=consume_data, args=(i, queue, args.batchsize))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for index in range(num_processes):
        summaries[index].report(
            batchsize=args.batchsize,
            output_prefix=None,
            display_all_summary=True,
            multi_threads=True)
        time.sleep(1)

