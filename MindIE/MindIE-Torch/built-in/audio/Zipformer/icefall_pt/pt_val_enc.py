# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
import copy
import argparse

import torch
import mindietorch
import numpy as np
from mindietorch import _enums
from torch.utils.data import dataloader

from model_pt_enc import forward_infer


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def get_dataloader(opt):
    x = torch.zeros(100, 80, dtype=torch.float32)
    x_lens = 100
    # x: input 0
    # x_lens: input 1
    datasets = [[copy.deepcopy(x), copy.deepcopy(x_lens)]]
    while len(datasets) % opt.batch_size != 0:
        datasets.append(datasets[-1])
    m = 1
    datasets_orig = copy.deepcopy(datasets)
    while m < opt.multi:
        datasets += datasets_orig
        m += 1

    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    print("OPT_BATCHSIZE: ", opt.batch_size)
    return loader(datasets,
                  batch_size=opt.batch_size,
                  shuffle=False,
                  num_workers=1,
                  sampler=None,
                  pin_memory=True)


def save_tensor_arr_to_file(arr, file_path):
    write_sen = ""
    for m in arr:
        for l in m:
            for c in l:
                write_sen += str(c) + " "
            write_sen += "\n"
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(write_sen)

def save_size_to_file(size, file_path):
    write_sen = "" + str(size) + " "
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(write_sen)

def main(opt):
    # load model
    model = torch.jit.load(opt.model)
    batch_size = opt.batch_size
    mindietorch.set_device(opt.device_id)
    if opt.need_compile:
        inputs = []
        inputs.append(mindietorch.Input([opt.batch_size, 100, 80], dtype=mindietorch.dtype.FLOAT))
        inputs.append(mindietorch.Input([opt.batch_size], dtype=mindietorch.dtype.INT32))

        model = mindietorch.compile(
            model,
            inputs=inputs,
            precision_policy=_enums.PrecisionPolicy.FP32,
            soc_version=opt.soc_version,
            optimization_level=0
        )

    dataloader = get_dataloader(opt)
    pred_results = forward_infer(model, dataloader, batch_size, opt.device_id)

    if opt.batch_size == 1 and opt.multi == 1:
        result_path = opt.result_path
        if(os.path.exists(result_path) == False):
            os.makedirs(result_path)
        for index, res in enumerate(pred_results):
            result_fname_0 = 'data' + str(index) + '_0.txt'
            result_fname_1 = 'data' + str(index) + '_1.txt'
            save_tensor_arr_to_file(np.array(res[0]), os.path.join(result_path, result_fname_0))
            save_size_to_file(res[1].numpy()[0], os.path.join(result_path, result_fname_1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zipformer offline model encoder inference.')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--model', type=str, default="./pt_compiled_model/encoder-epoch-12-avg-1_mindietorch_bs1.pt", help='ts model path')
    parser.add_argument('--need_compile', action="store_true", help='if the loaded model needs to be compiled or not')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--result_path', default='result/encoder')
    parser.add_argument('--multi', type=int, default=1, help='multiples of dataset replication for enough infer loop. if multi != 1, the pred result will not be stored.')
    opt = parser.parse_args()
    main(opt)
