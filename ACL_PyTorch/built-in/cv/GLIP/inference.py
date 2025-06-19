# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

import argparse
import os
import functools

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision.ops import boxes as box_ops

from ais_bench.infer.interface import InferSession
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.rpn.inference import make_atss_postprocessor
from maskrcnn_benchmark.modeling.rpn.vldyhead import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator_complex


if not hasattr(np, 'float'):
    np.float = np.float64


class postprocess(nn.Module):
    def __init__(self, cfg):
        super(postprocess, self).__init__()
        self.cfg = cfg
        box_coder = BoxCoder(cfg)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder, is_train=False)
        self.anchor_generator = make_anchor_generator_complex(cfg)

    def forward(self, images, features, box_regression, centerness, box_cls, dot_product_logits, positive_map):
        anchors = self.anchor_generator(images, features)
        boxes = self.box_selector_test(box_regression, centerness, anchors, \
                         box_cls, None, dot_product_logits, positive_map)
        return boxes

def patch_embedding(x):
    _, _, H, W = x.size()
    if W % 4 != 0:
        x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
    if H % 4 != 0:
        x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
    return x

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config-file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        default=None,
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    parser.add_argument(
        "--output_folder",
        default=None
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    data_loaders = make_data_loader(cfg, is_train=False)
    data_loader = data_loaders[0]
    dataset = data_loaders[0].dataset

    postprocess_model = postprocess(cfg)
    postprocess_model.to('cpu')
    checkpointer = DetectronCheckpointer(
        cfg, postprocess_model, save_dir=cfg.OUTPUT_DIR)
    if args.weight:
        _ = checkpointer.load(args.weight, force=True)
    postprocess_model.eval()

    device_id = 0

    model_backbone = "./backbone/model/glip_backbone_linux_aarch64.om"
    session_backbone = InferSession(device_id, model_backbone)
    model_rpn_fuse = "./rpn_head/model/glip_rpn_head_linux_aarch64.om"
    session_rpn_fuse = InferSession(device_id, model_rpn_fuse)    
    model_rpn_select = "./select/model/glip_select_linux_aarch64.om"
    session_rpn_select = InferSession(device_id, model_rpn_select)    


    all_output = []
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, *_ = batch
        input_image = patch_embedding(images.tensors)
        feed_backbone = [input_image]
        f1, f2, f3, f4, f5 = session_backbone.infer(
            feed_backbone, mode='dymshape', custom_sizes=20000000)
        feed_rpn_fuse = [f1, f2, f3, f4, f5]
        g1, g2, g3, g4, g5, g6 = session_rpn_fuse.infer(
            feed_rpn_fuse, mode='dymshape', custom_sizes=100000000)
        feeds_rpn_select = [g1, g2, g3, g4, g5, g6]
        o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, \
        o15, o16, o17, o18, o19, o20 = session_rpn_select.infer(
        feeds_rpn_select, mode='dymshape', custom_sizes=20000000)
        box_cls = [torch.from_numpy(i) for i in [o1,o2,o3,o4,o5]]
        box_regression = [torch.from_numpy(i) for i in [o6,o7,o8,o9,o10]]
        centerness = [torch.from_numpy(i) for i in [o11,o12,o13,o14,o15]]
        dot_product_logits = [torch.from_numpy(i) for i in [o16,o17,o18,o19,o20]]
        features = [torch.from_numpy(i) for i in [f1,f2,f3,f4,f5]]
        with torch.no_grad():
            positive_map = np.load('./rpn_head/positive_map.npy', allow_pickle=True).item()
            boxlist = postprocess_model(images, features, box_regression, centerness,
                                box_cls, dot_product_logits, positive_map)
        all_output.append(boxlist[0])
    predictions = all_output
    iou_types = ("bbox",)
    extra_args = dict(
        box_only=False,
        iou_types=iou_types,
        expected_results=[],
        expected_results_sigma_tol=4,
    )
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    return evaluate(dataset=dataset, predictions=predictions, output_folder=args.output_folder, **extra_args)

if __name__ == '__main__':
    main()