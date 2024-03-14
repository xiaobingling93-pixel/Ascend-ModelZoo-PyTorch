# Copyright 2023 Huawei Technologies Co., Ltd
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

_base_ = [
    '../configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py',
]

lang_model_name = '../bert-base-uncased/'
data_root = '../refcoco/'
load_from = "../weights/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"

model = dict(
    type='GroundingDINO',
    language_model=dict(
        name=lang_model_name,
        max_tokens=256,
        add_pooling_layer=False,
    ),
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        with_cp=False,
        patch_norm=True,
        convert_weights=True,
    ),
    neck=dict(in_channels=[256, 512, 1024]),
    encoder=dict(
        fusion_layer_cfg=dict(dropout=0.0)
    ),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=256,
        sync_cls_avg_factor=True,
        # contrastive_cfg=dict(max_text_len=256),
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)
    ),
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # change this
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file="mdetr_annotations/finetune_refcoco_train_vg.json",
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline))

# -------------------------------------------------#
ann_file = "mdetr_annotations/finetune_refcoco_val.json"
val_dataset_all_val = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=_base_.test_pipeline,
    backend_args=None)
val_evaluator_all_val = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcoco_testA.json'
val_dataset_refcoco_testA = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=_base_.test_pipeline,
    backend_args=None)

val_evaluator_refcoco_testA = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
ann_file = 'mdetr_annotations/finetune_refcoco_testB.json'
val_dataset_refcoco_testB = dict(
    type='MDETRStyleRefCocoDataset',
    data_root=data_root,
    ann_file=ann_file,
    data_prefix=dict(img='train2014/'),
    test_mode=True,
    return_classes=True,
    pipeline=_base_.test_pipeline,
    backend_args=None)

val_evaluator_refcoco_testB = dict(
    type='RefExpMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

# -------------------------------------------------#
datasets = [
    val_dataset_all_val,
    # val_dataset_refcoco_testA,
    # val_dataset_refcoco_testB
]
dataset_prefixes = [
    'refcoco_val',
    # 'refcoco_testA',
    # 'refcoco_testB'
]
metrics = [
    val_evaluator_all_val,
    # val_evaluator_refcoco_testA,
    # val_evaluator_refcoco_testB
]


val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

train_cfg = dict(_delete_=True, type="IterBasedTrainLoop", max_iters=2000, val_interval=200)


param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=2000,
        by_epoch=False,
        milestones=[1000],
        gamma=0.1)
]


auto_scale_lr = dict(base_batch_size=16)

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=1))

randomness = dict(seed=1234)

