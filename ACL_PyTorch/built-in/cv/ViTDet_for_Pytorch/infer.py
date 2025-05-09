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
import os
import time

import torch
import torch_npu
import torchair

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.amp import autocast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint file")
    parser.add_argument('--warm_up_times', default=2, type=int, help='warm up times')
    parser.add_argument('--loop', type=int, default=0, help="loop times")
    args = parser.parse_args()

    # load config
    cfg = Config.fromfile(args.cfg)
    cfg.load_from = args.ckpt
    # work_dir is required in runner initiator and is used to store temporary file
    cfg.work_dir = os.path.join('./work_dirs',
                                os.path.splitext(os.path.basename(args.cfg))[0])

    runner = Runner.from_cfg(cfg)

    runner.load_or_resume()
    config = torchair.CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    runner.model.data_preprocessor = torch.compile(runner.model.data_preprocessor,
                                                   backend=npu_backend,
                                                   fullgraph=False,
                                                   dynamic=True)
    runner.model.backbone = torch.compile(runner.model.backbone, backend=npu_backend, fullgraph=True)
    runner.model.neck = torch.compile(runner.model.neck, backend=npu_backend, fullgraph=True, dynamic=True)
    runner.model.rpn_head = torch.compile(runner.model.rpn_head, backend=npu_backend, fullgraph=True)
    runner.model.roi_head = torch.compile(runner.model.roi_head, backend=npu_backend, fullgraph=True)

    # warm up
    for _ in range(args.warm_up_times):
        runner.test()

    # loop==0 is recommended when infer with the whole dataset
    torch.npu.synchronize()
    start = time.time()
    for _ in range(args.loop):
        with torch.no_grad():
            for data_batch in runner.test_loop.dataloader:
                with autocast(enabled=runner.test_loop.fp16):
                    outputs = runner.model.test_step(data_batch)
                runner.test_loop.evaluator.process(data_samples=outputs, data_batch=data_batch)
                metrics = runner.test_loop.evaluator.evaluate(len(runner.test_loop.dataloader.dataset))
    torch.npu.synchronize()
    if args.loop > 0:
        print(f'E2E time = {(time.time() - start) / args.loop * 1000}ms')
