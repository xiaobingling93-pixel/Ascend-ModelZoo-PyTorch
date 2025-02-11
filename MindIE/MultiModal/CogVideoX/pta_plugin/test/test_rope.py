#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch.nn as nn
import torch_npu

torch.ops.load_library("../build/libPTAExtensionOPS.so")

if __name__ == "__main__":
    torch.npu.set_device(0)
    x = torch.randn((2, 48, 128, 64), device="npu")
    cos = torch.randn((1, 1, 128, 64), device="npu")
    sin = torch.randn((1, 1, 128, 64), device="npu")

    count = 5
    for i in range(count):
        output = torch.ops.mindie.rope_mindie_sd(x, cos, sin, mode=1)