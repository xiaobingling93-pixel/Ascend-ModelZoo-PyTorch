# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import argparse
import torch
import torch_npu
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="yolov11 dataset convert")
    parser.add_argument("--pth", type=str, help="weight")
    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--batchsize", type=int, help="batchsize")
    args = parser.parse_args()

    torch_npu.npu.set_compile_mode(jit_compile=False)
    model = YOLO(args.pth)
    with torch.no_grad():
        model.val(data=args.dataset, batch=args.batchsize, half=True, imgsz=640)