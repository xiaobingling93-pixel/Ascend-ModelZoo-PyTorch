# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import torch
from tqdm import tqdm

from MGN.opt import opt
from MGN.data import Data
from MGN.network import MGN
from MGN.utils.metrics import mean_ap, cmc, re_ranking

from om_executor import OMExcutor


def save_batch_images(save_file_name, dataset_type, loader, need_flip=False):
    index = 0
    for inputs, _ in loader:
        if need_flip is True:
            inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        for item in inputs:
            img_name = f"{index:05d}"
            save_path = opt.data_path
            if opt.data_path[-1] != '/':
                save_path += '/'
            save_path += save_file_name
            save_path = os.path.join(save_path, dataset_type)
            os.makedirs(save_path, exist_ok=True)
            bin_file_path = os.path.join(save_path, f"{img_name}.bin")
            item.numpy().tofile(bin_file_path)
            index += 1


class Preprocessor(OMExcutor):
    def __init__(self, data):
        super().__init__(data)

    def data_preprocess(self):
        file_name = 'bin_data'
        file_name_flip = 'bin_data_flip'
        save_batch_images(file_name, 'q', tqdm(self.query_loader))
        save_batch_images(file_name, 'g', tqdm(self.test_loader))
        save_batch_images(file_name_flip, 'q', tqdm(self.query_loader), need_flip=True)
        save_batch_images(file_name_flip, 'g', tqdm(self.test_loader), need_flip=True)


if __name__ == '__main__':
    data = Data()
    mgn_preprocessor = Preprocessor(data)
    print("start data preprocess")
    mgn_preprocessor.data_preprocess()