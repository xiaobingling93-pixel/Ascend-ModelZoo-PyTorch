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
from loguru import logger
import pypdfium2 as pdfium

import torch
import torch_npu
import torch.nn as nn
import torchvision
import torchvision_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

from mineru.backend.pipeline.model_list import AtomicModel
from mineru.model.mfr.unimernet.unimernet_hf.unimer_swin.modeling_unimer_swin import UnimerSwinSelfAttention
from mineru.backend.pipeline.model_init import (
    AtomModelSingleton,
    table_model_init,
    mfd_model_init,
    mfr_model_init,
    doclayout_yolo_model_init,
    ocr_model_init,
    )
from mineru.utils.model_utils import get_vram
from mineru.backend.pipeline.batch_analyze import (
    YOLO_LAYOUT_BASE_BATCH_SIZE,
    MFD_BASE_BATCH_SIZE,
    MFR_BASE_BATCH_SIZE,
    )

from transformers.generation.utils import GenerationMixin
from MinerU.demo.demo import parse_doc


def parse_args():
    parser = argparse.ArgumentParser("MinerU infer")
    parser.add_argument("--model_source", type=str, default="local", help="model checkpoint source")
    parser.add_argument("--data_path", type=str, default="OmniDocBench_dataset")
    parser.add_argument("--warmup", type=int, default=2, help="Warm up times")
    parser.add_argument("--warmup_data_path", type=str, default="OmniDocBench_dataset/pdfs/jiaocai_71434495.pdf_0.pdf")
    args = parser.parse_args()
    return args


def atom_model_init_compile(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        atom_model = doclayout_yolo_model_init(
            kwargs.get('doclayout_yolo_weights'),
            kwargs.get('device')
        )
        atom_model.model.model = compile_model(atom_model.model.model, False, True)
        npu_input = torch.zeros((batch_candidate[AtomicModel.Layout][0], 3, atom_model.imgsz, atom_model.imgsz))
        tng.inference.set_dim_gears(npu_input, {0: batch_candidate[AtomicModel.Layout]})

    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'),
            kwargs.get('device')
        )
        atom_model.model.model = compile_model(atom_model.model.model, False, True)
        npu_input = torch.zeros((batch_candidate[AtomicModel.MFD][0], 3, atom_model.imgsz, atom_model.imgsz))
        tng.inference.set_dim_gears(npu_input, {0: batch_candidate[AtomicModel.MFD]})

    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get('mfr_weight_dir'),
            kwargs.get('device')
        )

        modify_mfr_model(atom_model.model)

        atom_model.model.encoder = compile_model(atom_model.model.encoder, False, True)
        atom_model.model.decoder = compile_model(atom_model.model.decoder, True, True)
        
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get('det_db_box_thresh'),
            kwargs.get('lang'),
            kwargs.get('det_limit_side_len'),
        )

    elif model_name == AtomicModel.Table:
        atom_model = table_model_init(
            kwargs.get('lang'),
        )
        
    else:
        logger.error('model name not allow')
        raise ValueError("model name not allow")

    if atom_model is None:
        logger.error('model init failed')
        raise RuntimeError("model init failed")

    return atom_model


def rewrite_mfr_encoder_multi_head_attention_forward(model):
    wq = model.query.weight
    wk = model.key.weight
    wv = model.value.weight
    model.qkv = nn.Linear(in_features=wk.shape[1], out_features=wq.shape[0] + wk.shape[0] + wv.shape[0])
    model.qkv.weight = nn.Parameter(torch.concat([wq, wk, wv], dim=0), requires_grad=False)
    wq_bias = model.query.bias if model.query.bias is not None else torch.zeros(wq.shape[0])
    wk_bias = model.key.bias if model.key.bias is not None else torch.zeros(wk.shape[0])
    wv_bias = model.key.bias if model.value.bias is not None else torch.zeros(wv.shape[0])
    model.qkv.bias = nn.Parameter(torch.concat([wq_bias, wk_bias, wv_bias], dim=0), requires_grad=False)


def modify_mfr_model(model):
    # 修改encoder的attention forward
    for _, module in model.encoder.named_modules():
        if isinstance(module, UnimerSwinSelfAttention):
            rewrite_mfr_encoder_multi_head_attention_forward(module)
    rewrite_mfr_encoder_forward()


def compile_model(model, dynamic, fullgraph):
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    compiled_model = torch.compile(model, dynamic=dynamic, fullgraph=fullgraph, backend=npu_backend)
    return compiled_model


def rewrite_model_init():
    def _patched_getmodel(self, atom_model_name: str, **kwargs):
        lang = kwargs.get('lang', None)
        table_model_name = kwargs.get('table_model_name', None)

        if atom_model_name in [AtomicModel.OCR]:
            key = (atom_model_name, lang)
        elif atom_model_name in [AtomicModel.Table]:
            key = (atom_model_name, table_model_name, lang)
        else:
            key = atom_model_name

        if key not in self._models:
            self._models[key] = atom_model_init_compile(model_name=atom_model_name, **kwargs)
        return self._models[key]
    AtomModelSingleton.get_atom_model = _patched_getmodel


def rewrite_mfr_encoder_forward():
    def _patched_prepare_encoder_decoder_kwargs_for_generation(self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name,
        generation_config,
    ):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs and generation config.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value 
                for argument, value in encoder_kwargs.items() 
                if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True

        ####### 固定input_tensor形状
        pad_count = 0
        if batch_candidate[AtomicModel.MFR] != inputs_tensor.shape[0]:
            pad_count = batch_candidate[AtomicModel.MFR] - inputs_tensor.shape[0]
            padding_tensor = torch.zeros(pad_count, *inputs_tensor.shape[1:], dtype=inputs_tensor.dtype, device=inputs_tensor.device)
            inputs_tensor = torch.cat((inputs_tensor, padding_tensor), dim=0)

        encoder_kwargs[model_input_name] = inputs_tensor
        output = encoder(**encoder_kwargs)# type: ignore
        if pad_count != 0:
            output.last_hidden_state = output.last_hidden_state[:-pad_count]
            output.pooler_output = output.pooler_output[:-pad_count]
        model_kwargs["encoder_outputs"] = output
        return model_kwargs

    GenerationMixin._prepare_encoder_decoder_kwargs_for_generation = _patched_prepare_encoder_decoder_kwargs_for_generation


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
        

def get_pdf_page_count(pdf_path):
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()


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
    print(len(doc_path_list), batch_candidate)
    warmup(args.warmup_data_path, args.warmup)
    
    print("******** 精度测试 **********")
    start_time = time.time()
    parse_doc(doc_path_list, output_dir, backend="pipeline")
    print(f"per page process time: {(time.time()-start_time)/pdfs_page_count:.2f}s")
