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
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from funasr import AutoModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sensevoice infer")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument('--input', type=str, help='input audio file')
    parser.add_argument('--loop', default=10, type=int, help='loop time')
    args = parser.parse_args()

    # 初始化模型
    model = AutoModel(model=args.model_path, trust_remote_code=True, device="npu", fp16=True, disable_update=True)

    # 开启图模式
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    model.model.encoder = torch.compile(model.model.encoder, dynamic=True, fullgraph=True, backend=npu_backend)

    print('warm up start')
    res = model.generate(
        input=f"{args.model_path}/example/zh.mp3",
        cache={},
        language="auto",
        use_itn=True,
    )
    print('warm up end')
    print(f"输出结果为:{res}")
    print('性能测试：')
    for i in range(args.loop):
        res = model.generate(
            input=f"{args.model_path}/example/zh.mp3",
            cache={},
            language="auto",
            use_itn=True,
        )