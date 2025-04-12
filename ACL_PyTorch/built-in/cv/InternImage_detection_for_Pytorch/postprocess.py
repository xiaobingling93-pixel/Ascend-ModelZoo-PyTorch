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

import torch
from mmengine.config import Config
from mmdet.apis import init_detector
from mmengine.registry import Registry, build_from_cfg
from mmdet.registry import DATASETS
from tqdm import tqdm
import numpy as np
from mmengine.registry import EVALUATOR
from mmdet.evaluation import CocoMetric

from preprocess import parse_shape, adjust_cfg
import mmdet_custom  # noqa: F401,F403

# for each img input, 8 NPY files will be output
NUM_OM_OUTPUT_FILE = 8
# the output NPY files with postfix 2-6 are feature map from FPN
FEAT_IDX_START = 2
FEAT_IDX_END = 7


def process_batch(cfg, om_output_path, evaluator, basename_metas, model=None):
    for filename in tqdm(basename_metas.keys(), desc='post-processing'):
        img_meta = basename_metas.get(filename)
        try:
            # load all the om output files for post-processing
            file_path_prefix = os.path.join(om_output_path, filename)
            cls_scores = np.load(f'{file_path_prefix}_0.npy')
            bbox_preds = np.load(f'{file_path_prefix}_1.npy')
            feature_map = []
            for i in range(FEAT_IDX_START, FEAT_IDX_END):
                feature_map.append(np.load(f'{file_path_prefix}_{i}.npy'))
            rois = np.load(f'{file_path_prefix}_7.npy')

            cls_scores = torch.from_numpy(cls_scores)
            bbox_preds = torch.from_numpy(bbox_preds)
            feature_map = [torch.from_numpy(fm) for fm in feature_map]
            rois = torch.from_numpy(rois)

            bbox_results = model.roi_head.bbox_head[-1].predict_by_feat(
                rois=[rois],
                cls_scores=[cls_scores],
                bbox_preds=[bbox_preds],
                batch_img_metas=[img_meta],
                rescale=False,
                rcnn_test_cfg=cfg.model.test_cfg.rcnn)
            mask_results = model.roi_head.predict_mask(
                feature_map, [img_meta], bbox_results, rescale=True)

            # construct the legal input for evaluator
            result = {
                'pred_instances': {
                    'bboxes': mask_results[0]['bboxes'],
                    'scores': mask_results[0]['scores'],
                    'labels': mask_results[0]['labels'],
                    'masks': mask_results[0]['masks']
                },
                **img_meta
            }
            evaluator.process(data_batch={}, data_samples=[result])
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="post process data")
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='ckpt file path')
    parser.add_argument('--om_output', type=str, required=True, help='om output for post-process')
    parser.add_argument(
        '--force_img_shape', type=parse_shape, default=None, help='Rescale image to shape (e.g., "256,256")')
    parser.add_argument('--batch_size', type=int, default=100, help='number of processed imgs at the same time')
    parser.add_argument('--eval', nargs='+', type=str, help='evaluation types, e.g., bbox, segm')
    args = parser.parse_args()

    cfg = adjust_cfg(Config.fromfile(args.config), force_img_shape=args.force_img_shape)
    dataset = build_from_cfg(cfg.data.test, DATASETS, None)

    basename_metas = {}
    for data in tqdm(dataset, desc='loading metainfo'):
        img_meta = data['data_sample'][0].metainfo
        basename = os.path.basename(img_meta['img_path']).split('.')[0]
        basename_metas[basename] = img_meta

    model = init_detector(args.config, args.ckpt, device='cpu')

    # construct evaluator
    eval_cfg = dict(
        type='mmdet.evaluation.CocoMetric',
        ann_file=dataset.ann_file,
        metric=args.eval,
        classwise=True,
        format_only=False,
        _scope_='mmdet.evaluation',
    )
    eval_cfg.update(cfg.get('evaluation', {}))
    evaluator = EVALUATOR.build(eval_cfg)
    evaluator.dataset_meta = dataset.metainfo

    print('Start post-processing')
    process_batch(cfg, args.om_output, evaluator, basename_metas, model)

    print('Evaluating final results')
    metrics = evaluator.evaluate(size=len(dataset))
    print(f"metric: bbox_mAP: {metrics['coco/bbox_mAP']}, segm_mAP: {metrics['coco/segm_mAP']}")