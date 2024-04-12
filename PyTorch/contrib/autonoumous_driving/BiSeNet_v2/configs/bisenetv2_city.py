# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=19,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=150000,
    dataset='CityScapes',
    im_root='./datasets/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    scales=[0.25, 2.],
    cropsize=[512, 1024],
    eval_crop=[1024, 1024],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=16,
    num_workers=32,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
