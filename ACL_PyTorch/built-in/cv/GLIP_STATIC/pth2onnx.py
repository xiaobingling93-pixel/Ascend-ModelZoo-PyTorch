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
import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector.onnx_model import SWIN_BACKBONE, FUSE_MODEL, LANG


def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config-file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        help="pth to model",
        default="glip_tiny_model_o365_goldg.pth"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        "--model_type",
        choices=["lang","rpn_head","backbone"],
        help="convert model type",
    )

    args = parser.parse_args()    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  
    if args.model_type == "backbone":
        model = SWIN_BACKBONE(cfg)
    elif args.model_type == "rpn_head":
        model = FUSE_MODEL(cfg)
    elif args.model_type == "lang":
        model = LANG(cfg)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(args.weight, force=True)
    iou_types = ("bbox",)
    model.eval()
    if args.model_type == "backbone":
        image = torch.rand(1,3,784,1344) #指定输入image shape
        torch.onnx.export(model, image, './glip_backbone.onnx', input_names=['images'],\
                    output_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'], opset_version=11)
    elif args.model_type == "rpn_head":
        #指定输入特征图shape大小
        feature_1 = torch.rand(1,256,98,168)
        feature_2 = torch.rand(1,256,49,84)
        feature_3 = torch.rand(1,256,25,42)
        feature_4 = torch.rand(1,256,13,21)
        feature_5 = torch.rand(1,256,7,11)
        #指定输入文本特征长度
        lang = torch.rand(1,256, 768)
        mask = torch.rand(1,256)
        dummy_input = (feature_1, feature_2, feature_3, feature_4, feature_5, lang, mask)
        torch.onnx.export(model, dummy_input, './glip_rpn.onnx', input_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'lang', 'mask'],\
                        output_names=['o1', 'o2', 'o3', 'o4', 'o5', 'o6','o7', 'o8', 'o9', 'o10', 'o11', 'o12','o13', 'o14', 'o15'], opset_version=11)
    elif args.model_type == "lang":
        # 制定文本长度为256
        lang = torch.randint(1,500,(1,256))
        mask = torch.zeros(1,256)
        dummy_input = (lang, mask)
        torch.onnx.export(model, dummy_input, './glip_language.onnx', input_names=['lang', 'mask'],\
                    output_names=['lang', 'mask'], opset_version=11)


if __name__ == '__main__':
    main()