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
        convert_weights=True,
    ),
    neck=dict(in_channels=[256, 512, 1024]),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=256,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
    )
)


test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, return_classes=True))
test_dataloader = val_dataloader
