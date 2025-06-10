#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""

import torch
if torch.__version__ >= "2.6":
    torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

from fairseq_cli.train import cli_main


if __name__ == "__main__":
    cli_main()
