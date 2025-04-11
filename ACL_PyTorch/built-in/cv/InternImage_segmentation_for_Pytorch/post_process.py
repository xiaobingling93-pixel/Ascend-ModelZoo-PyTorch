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
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from mmengine.config import Config
from mmengine.registry import Registry, build_from_cfg

from mmseg.registry import DATASETS
from mmseg.datasets.ade import ADE20KDataset
from mmseg.datasets.transforms import ResizeToMultiple
from mmseg.evaluation import IoUMetric


def load_gt_seg_map(gt_seg_map_path, reduce_zero_label=False):
    gt = cv2.imread(gt_seg_map_path, cv2.IMREAD_GRAYSCALE)
    if reduce_zero_label:
        gt[gt == 0] = 255
        gt = gt - 1
        gt[gt == 254] = 255
    return gt.astype(np.uint8)


def process_npy_files(input_dir, dataset):
    npy_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]

    data_dict = dict()

    print('\033[92m' + 'loading dataset' + '\033[0m')
    for data in tqdm(dataset):
        img_metas = data['data_sample'][0].metainfo
        k = os.path.basename(img_metas['img_path'])
        data_dict[k] = img_metas
    print('\033[92m' + 'dataset loaded' + '\033[0m')

    seg_map_paths = []
    dataset_meta = dict(
        classes=dataset.metainfo['classes'],
        palette=dataset.metainfo['palette']
    )
    metric = IoUMetric(
        iou_metrics=['mIoU'],
        ignore_index=dataset.ignore_index,
        output_dir=None,
        format_only=False
    )
    metric.dataset_meta = dataset_meta

    for npy_file in tqdm(npy_files):
        data = torch.from_numpy(np.load(npy_file))
        target_img_meta_path = npy_file.rsplit('_', 1)[0] + '.jpg'
        target_img_meta_name = os.path.basename(target_img_meta_path)

        img_meta = data_dict.get(target_img_meta_name)

        if img_meta is None:
            raise KeyError(f"corresponding image meta {target_img_meta_name} not found")

        if data.shape[0] != 1 or data.shape[1] != 150:
            raise ValueError(f"unexpected shape {data.shape}")
        else:
            img_shape = img_meta['img_shape'][:2]
            data = F.interpolate(data, img_shape, None, 'bilinear', False)

            # remove padding
            resize_shape = img_meta['img_shape'][:2]
            data = data[:, :, :resize_shape[0], :resize_shape[1]]

            size = img_meta['ori_shape'][:2]
            data = F.interpolate(data, size, None, 'bilinear', align_corners=False)
            output = F.softmax(data, dim=1)
            seg_pred = output.argmax(dim=1)
            seg_map_paths.append(img_meta['seg_map_path'])
            seg_map_path = img_meta['seg_map_path']
            gt_seg_map = load_gt_seg_map(seg_map_path, dataset.reduce_zero_label)
            gt_seg_map = torch.from_numpy(gt_seg_map).unsqueeze(0)
            data_sample = {
                'pred_sem_seg': {
                    'data': seg_pred
                },
                'gt_sem_seg': {
                    'data': gt_seg_map
                }
            }
            metric.process(gt_seg_map, [data_sample])

    metrics = metric.compute_metrics(metric.results)
    return metrics


def main(config_path, om_output_path):
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # config modification made for the update from mmseg v0.17.0 to v1.2.2
    cfg.data.test['data_prefix'] = {
        'img_path': cfg.data.test.pop('img_dir'),
        'seg_map_path': cfg.data.test.pop('ann_dir')
    }

    dataset = build_from_cfg(cfg.data.test, DATASETS, None)
    metric = process_npy_files(om_output_path, dataset)
    print(metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="post process data")
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--input', type=str, required=True, help='input of post-process script')
    args = parser.parse_args()

    main(args.config, args.input)
