# Copyright 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import argparse

import mmcv
import torch
from mmengine.config import Config
import numpy as np
from tqdm import tqdm
from mmengine.registry import Registry, build_from_cfg

from mmseg.registry import DATASETS
from mmseg.datasets.ade import ADE20KDataset
from mmseg.datasets.transforms import ResizeToMultiple


def main(config_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # config modification made for the update from mmseg v0.17.0 to v1.2.2
    cfg.data.test['data_prefix'] = {
        'img_path': cfg.data.test.pop('img_dir'),
        'seg_map_path': cfg.data.test.pop('ann_dir')
    }

    dataset = build_from_cfg(cfg.data.test, DATASETS, None)

    for data in tqdm(dataset):
        img = data['inputs'][0].unsqueeze(0)
        file_path = data['data_sample'][0].metainfo['img_path']
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.npy')
        np.save(output_path, img)

    print('\033[92m' + 'data preprocessing finished' + '\033[0m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument('--config', type=str, required=True, help='config path')
    parser.add_argument('--output', type=str, required=True, help='output path for preprocessed data')
    args = parser.parse_args()

    config_path, output_dir = args.config, args.output
    main(config_path, output_dir)
