# Copyright 2025 Huawei Technologies Co., Ltd
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
from mmengine.config import Config
from mmengine.registry import Registry, build_from_cfg
from mmdet.registry import DATASETS
from tqdm import tqdm
import numpy as np


def preprocess_data(dataset, data_output_dir, force_img_shape=None, img_shape_output_dir="./img_shape"):
    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(img_shape_output_dir, exist_ok=True)

    for data in tqdm(dataset):
        img = data['inputs'][0].unsqueeze(0)
        file_path = data['data_sample'][0].metainfo['img_path']
        new_filename = os.path.splitext(os.path.basename(file_path))[0] + '.npy'
        data_output_path = os.path.join(data_output_dir, new_filename)
        img_shape_output_path = os.path.join(img_shape_output_dir, new_filename)
        np.save(data_output_path, img)
        if force_img_shape:
            np.save(img_shape_output_path, force_img_shape)
        else:
            np.save(img_shape_output_path, data['data_sample'][0].metainfo['img_shape'][:2])


def parse_shape(s):
    try:
        return [int(x) for x in s.split(',')]
    except Exception as e:
        raise argparse.ArgumentTypeError("Shape must be 'width,height' (e.g., '256,256')") from e


def adjust_cfg(cfg: dict, force_img_shape=None):
    img_norm_cfg = cfg.img_norm_cfg
    if force_img_shape:
        scale = tuple(force_img_shape)
        keep_ratio = False
    else:
        scale = cfg.data.test.pipeline[1].transforms[0].scale
        keep_ratio = cfg.data.test.pipeline[1].transforms[0].keep_ratio
    cfg.data.test.pipeline[1] = dict(type='MultiScaleFlipAug',
                                     transforms=[
                                         dict(type='Resize', keep_ratio=keep_ratio, scale=scale),
                                         dict(type='Normalize', **img_norm_cfg),
                                         dict(type='mmdet.PackDetInputs',
                                              meta_keys=(
                                                  'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
                                     ])
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument('--config', type=str, required=True, help='config path')
    parser.add_argument('--data_output', type=str, required=True, help='output path for preprocessed data')
    parser.add_argument(
        '--force_img_shape', type=parse_shape, help='Rescale image to shape (e.g., "256,256")')
    parser.add_argument('--img_shape_output', type=str, default="./img_shape",
                        help='output path for preprocessed img shape')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg = adjust_cfg(cfg, args.force_img_shape)
    dataset = build_from_cfg(cfg.data.test, DATASETS, None)
    preprocess_data(dataset, args.data_output, args.force_img_shape, args.img_shape_output)
    print('\033[92m' + 'data preprocessing finished' + '\033[0m')