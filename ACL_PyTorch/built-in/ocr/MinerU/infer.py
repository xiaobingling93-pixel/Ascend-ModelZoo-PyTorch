# Copyright 2025 Huawei Technologies Co., Ltd
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

import os
import sys
import math
import time 
import inspect

from pathlib import Path
import argparse

from mineru.backend.pipeline.model_list import AtomicModel
from mineru.utils.model_utils import get_vram
from mineru.utils.torchair_utils import (
    get_pdf_page_count,
    rewrite_model_init,
    set_batch_candidate,
    )
from mineru.backend.pipeline.batch_analyze import (
    YOLO_LAYOUT_BASE_BATCH_SIZE,
    MFD_BASE_BATCH_SIZE,
    MFR_BASE_BATCH_SIZE,
    )

from MinerU.demo.demo import parse_doc


def parse_args():
    parser = argparse.ArgumentParser("MinerU infer")
    parser.add_argument("--model_source", type=str, default="local", help="model checkpoint source")
    parser.add_argument("--data_path", type=str, default="OmniDocBench_dataset")
    parser.add_argument("--warmup", type=int, default=2, help="Warm up times")
    parser.add_argument("--warmup_data_path", type=str, default="OmniDocBench_dataset/pdfs/jiaocai_71434495.pdf_0.pdf")
    args = parser.parse_args()
    return args


def warmup(data_path, warmup_iters):
    data_path = Path(data_path)

    output_dir = Path(data_path).parent
    output_dir = os.path.join(output_dir, "warmup_res")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]
    supported_suffixes = pdf_suffixes + image_suffixes

    if data_path.suffix.lower() not in supported_suffixes:
        raise ValueError(
            f"Unsupported file type: '{data_path.suffix}'. "
            f"Supported types: {supported_suffixes}"
        )

    doc_path_list = [data_path] * sum(batch_candidate[AtomicModel.Layout])
    for _ in range(warmup_iters):
        parse_doc(doc_path_list, output_dir, backend="pipeline")

if __name__ == '__main__':
    args = parse_args()
    os.environ['MINERU_MODEL_SOURCE'] = args.model_source

    __dir__ = args.data_path
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    output_dir = os.path.join(__dir__, "output")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]


    print(pdf_files_dir)
    batch_ratio = 16

    rewrite_model_init()

    doc_path_list = []
    pdfs_page_count = 0
    for doc_path in Path(pdf_files_dir).glob('*'):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)
            pdfs_page_count += get_pdf_page_count(doc_path)

    batch_candidate = {
        AtomicModel.Layout: [YOLO_LAYOUT_BASE_BATCH_SIZE, pdfs_page_count % YOLO_LAYOUT_BASE_BATCH_SIZE],
        AtomicModel.MFD: [MFD_BASE_BATCH_SIZE, pdfs_page_count % MFD_BASE_BATCH_SIZE],
        AtomicModel.MFR: batch_ratio * MFR_BASE_BATCH_SIZE,
    }
    set_batch_candidate(batch_candidate)
    print(len(doc_path_list), batch_candidate)
    warmup(args.warmup_data_path, args.warmup)

    print("******** 精度测试 **********")
    start_time = time.time()
    parse_doc(doc_path_list, output_dir, backend="pipeline")
    print(f"per page process time: {(time.time()-start_time)/pdfs_page_count:.2f}s")
