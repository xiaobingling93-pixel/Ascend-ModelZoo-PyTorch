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

# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


import time
import logging
from tqdm import tqdm
import torch
import torch_npu
import acl
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

from funasr.auto.auto_model import AutoModel, tables, deep_update, prepare_data_iterator


class TorchairAutoModel(AutoModel):
    def __init__(self, **kwargs):
        log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
        logging.basicConfig(level=log_level)

        if not kwargs.get("disable_log", True):
            tables.print()
        
        model, kwargs = self.build_model(**kwargs)
        # fp16 for Atlas 300I DUO, bf16 for Atlas 800I A2
        # if the model is contextual paraformer for hotword inference,
        # must use fp32
        soc_version = acl.get_soc_name()
        kwargs['soc_version'] = soc_version
        if "910" in soc_version:
            kwargs["bf16"] = True
            kwargs["use_flash_attention"] = True
            model.to(torch.bfloat16)
        elif "310" in soc_version and kwargs['model'] == 'BiCifParaformer':
            kwargs["fp16"] = True
            model.to(torch.float16)
        threshold = kwargs.get("cif_threshold", None)
        if threshold is not None:
            model.predictor.threshold = threshold

        # if vad_model is not None, build vad model else None
        vad_model = kwargs.get("vad_model", None)
        vad_kwargs = {} if kwargs.get("vad_kwargs", {}) is None else kwargs.get("vad_kwargs", {})
        if vad_model is not None:
            logging.info("Building VAD model")
            vad_kwargs["model"] = vad_model
            vad_kwargs["model_revision"] = kwargs.get("vad_model_revision", "master")
            vad_kwargs["device"] = kwargs["device"]
            vad_model, vad_kwargs = self.build_model(**vad_kwargs)
        
        # if punc_model is not None, build punc model else None
        punc_model = kwargs.get("punc_model", None)
        punc_kwargs = {} if kwargs.get("punc_kwargs", {}) is None else kwargs.get("punc_kwargs", {})
        if punc_model is not None:
            logging.info("Building punc model.")
            punc_kwargs["model"] = punc_model
            punc_kwargs["model_revision"] = kwargs.get("punc_model_revision", "master")
            punc_kwargs["device"] = kwargs["device"]
            punc_model, punc_kwargs = self.build_model(**punc_kwargs)
        
        torch_npu.npu.set_compile_mode(jit_compile=False)
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=config)

        print("compile model...")
        compile_encoder = not ("310" in soc_version and kwargs['model'] == 'ContextualParaformer')
        if compile_encoder:
            model.encoder = torch.compile(model.encoder, dynamic=True, fullgraph=True, backend=npu_backend)
            tng.use_internal_format_weight(model.encoder)
        model.decoder = torch.compile(model.decoder, dynamic=True, fullgraph=True, backend=npu_backend)
        tng.use_internal_format_weight(model.decoder)
        if hasattr(model.predictor, "process_hidden") and callable(getattr(model.predictor, "process_hidden")):
            model.predictor.process_hidden = torch.compile(model.predictor.process_hidden, dynamic=True, fullgraph=True, backend=npu_backend)

        self.kwargs = kwargs
        self.model = model
        self.vad_model = vad_model
        self.vad_kwargs = vad_kwargs
        self.punc_model = punc_model
        self.punc_kwargs = punc_kwargs
        self.spk_model = None
        self.spk_kwargs = {}
        self.model_path = kwargs.get("model_path")
    

    def generate(self, input_data, input_len=None, progress_callback=None, **cfg):
        if self.vad_model is None:
            return self.inference_with_asr(
                input_data, input_len=input_len, progress_callback=progress_callback, **cfg
            )
        else:
            return self.inference_with_vad(
                input_data, input_len=input_len, progress_callback=progress_callback, **cfg
            )


    def pad_inputs(self, key_list, data_list, batch_size):
        need_pad_num = batch_size - len(data_list) % batch_size
        for _ in range(need_pad_num):
            key_list.append(key_list[-1])
            data_list.append(data_list[-1])
        return key_list, data_list


    def inference_with_asr(self, input_data, input_len=None, model=None, kwargs=None, key=None, display_pbar=False, **cfg):
        kwargs = self.kwargs if kwargs is None else kwargs
        deep_update(kwargs, cfg)
        model = self.model if model is None else model
        model.eval()

        batch_size = kwargs.get("batch_size", 1)

        key_list, data_list = prepare_data_iterator(
            input_data, input_len=input_len, data_type=kwargs.get("data_type", None), key=key
        )

        if len(key_list) % batch_size != 0:
            key_list, data_list = self.pad_inputs(key_list, data_list, batch_size)
        
        time_stats = {"avg_trans_rate": 0.0, "input_speech_time": 0.0, "end_to_end_time": 0.0, "pure_infer_time": 0.0,
                      "load_data": 0.0}
        asr_result_list = []
        num_samples = len(data_list)

        if display_pbar:
            pbar = tqdm(colour="blue", total=num_samples, dynamic_ncols=True)
        
        for beg_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, beg_idx + batch_size)
            data_batch = data_list[beg_idx:end_idx]
            key_batch = key_list[beg_idx:end_idx]
            batch = {"data_in": data_batch, "key": key_batch}

            if (end_idx - beg_idx) == 1 and kwargs.get("data_type", None) == "fbank": # fbank
                batch["data_n"] = data_batch[0]
                batch["data_lengths"] = input_len
            
            with torch.no_grad():
                time1 = time.perf_counter()
                res = model.inference(**batch, **kwargs)
                time2 = time.perf_counter()
                if isinstance(res, (list, tuple)):
                    results = res[0] if len(res) > 0 else [{"text": ""}]
                    meta_data = res[1] if len(res) > 1 else {}
            
            asr_result_list.extend(results)

            batch_data_time = meta_data.get("batch_data_time", 0.0)
            time_escape = time2 - time1

            time_stats["load_data"] += float(meta_data.get("load_data", 0.0))
            time_stats["end_to_end_time"] += time_escape
            time_stats["input_speech_time"] += batch_data_time
            time_stats["pure_infer_time"] = time_stats["end_to_end_time"] - time_stats["load_data"]
            time_stats["avg_trans_rate"] = time_stats["input_speech_time"] / time_stats["pure_infer_time"]

            if display_pbar:
                pbar.update(batch_size)
                pbar.set_description("avf_trans_rate:{:.3f}".format(time_stats["avg_trans_rate"]))
        
        return asr_result_list, time_stats
