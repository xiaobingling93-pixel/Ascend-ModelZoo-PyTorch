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

from ais_bench.infer.interface import InferSession
from maskrcnn_benchmark.modeling.detector.onnx_model import SWIN_BACKBONE, FUSE_MODEL, LANG
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

def om_infer(lang, mask, image):
    device_id = 0
    #加载om模型
    model_backbone = "./glip_backbone.om"
    session_backbone = InferSession(device_id, model_backbone)
    model_rpn = "./glip_rpn.om"
    session_rpn = InferSession(device_id, model_rpn)    
    model_lang = "./glip_language.om"
    session_lang = InferSession(device_id, model_lang)
    #加载输入，执行推理
    feed_backbone = [image]
    feed_lang = [lang, mask]
    f1, f2, f3, f4, f5 = session_backbone.infer(feed_backbone)
    l1, l2 = session_lang.infer(feed_lang)
    feeds_rpn = [f1, f2, f3, f4, f5, l1, l2]
    out = session_rpn.infer(feeds_rpn)
    return out

def cpu_infer(cfg, weight, lang, mask, image):
    #初始化torch模型
    model_backbone = SWIN_BACKBONE(cfg)
    checkpointer = DetectronCheckpointer(cfg, model_backbone)
    _ = checkpointer.load(weight, force=True)
    model_backbone.eval()
    model_rpn = FUSE_MODEL(cfg)
    checkpointer = DetectronCheckpointer(cfg, model_rpn)
    _ = checkpointer.load(weight, force=True)
    model_rpn.eval()
    model_lang = LANG(cfg)
    checkpointer = DetectronCheckpointer(cfg, model_lang)
    _ = checkpointer.load(weight, force=True)
    model_lang.eval()
    #执行cpu推理
    with torch.no_grad():
        f1, f2, f3, f4, f5 = model_backbone(image)
        l1, l2 = model_lang(lang,mask)
        out = model_rpn(f1, f2, f3, f4, f5, l1, l2)
    return out

def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config_file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        help="pth to model",
        default="glip_tiny_model_o365_goldg.pth"
    )
    
    args = parser.parse_args()    
    cfg.merge_from_file(args.config_file)
    cfg.freeze() 
    #构造随机输入数据
    lang = torch.randint(1,500,(1,256))
    mask = torch.ones(1,256)
    image = torch.rand(1,3,784,1344)
    # cpu推理
    x = cpu_infer(cfg, args.weight, lang, mask, image)
    # om推理
    y = om_infer(lang, mask, image)
    # 计算余弦相似度
    for i in range(15):
        similarity = torch.cosine_similarity(x[i].reshape(-1), torch.from_numpy(y[i]).reshape(-1), dim=0)
        print(f"第{i}个输出的余弦相似度为:")
        print(similarity)
    
if __name__ == '__main__':
    main()