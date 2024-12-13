# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append("./FunASR")

import time
import logging

import torch
from tqdm import tqdm

from mindie_paraformer import MindieBiCifParaformer
from mindie_encoder_decoder import MindieEncoder, MindieDecoder
from mindie_punc import MindiePunc, MindieCTTransformer
from mindie_cif import MindieCifTimestamp, MindieCif
from mindie_vad import MindieVAD
from funasr.auto.auto_model import AutoModel, download_model, tables, deep_update, \
    load_pretrained_model, prepare_data_iterator


class MindieAutoModel(AutoModel):
    def __init__(self, **kwargs):
        log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
        logging.basicConfig(level=log_level)

        if not kwargs.get("disable_log", True):
            tables.print()

        kwargs["compile_type"] = "paraformer"
        model, kwargs = self.build_model_with_mindie(**kwargs)

        # if vad_model is not None, build vad model else None
        vad_model = kwargs.get("vad_model", None)
        vad_kwargs = {} if kwargs.get("vad_kwargs", {}) is None else kwargs.get("vad_kwargs", {})
        if vad_model is not None:
            logging.info("Building VAD model.")
            vad_kwargs["model"] = vad_model
            vad_kwargs["model_revision"] = kwargs.get("vad_model_revision", "master")
            vad_model, vad_kwargs = self.build_model(**vad_kwargs)
            vad_kwargs["device"] = "npu"
            compiled_vad = torch.jit.load(kwargs["compiled_vad"])
            vad_model.encoder = compiled_vad

        # if punc_model is not None, build punc model else None
        punc_model = kwargs.get("punc_model", None)
        punc_kwargs = {} if kwargs.get("punc_kwargs", {}) is None else kwargs.get("punc_kwargs", {})
        if punc_model is not None:
            logging.info("Building punc model.")
            punc_kwargs["model"] = punc_model
            punc_kwargs["model_revision"] = kwargs.get("punc_model_revision", "master")
            punc_kwargs["device"] = "cpu"
            punc_kwargs["compile_type"] = "punc"
            punc_kwargs["compiled_punc"] = kwargs["compiled_punc"]
            punc_model, punc_kwargs = self.build_model_with_mindie(**punc_kwargs)

        self.kwargs = kwargs
        self.model = model
        self.vad_model = vad_model
        self.vad_kwargs = vad_kwargs
        self.punc_model = punc_model
        self.punc_kwargs = punc_kwargs
        self.spk_model = None
        self.spk_kwargs = {}
        self.model_path = kwargs.get("model_path")

    def generate(self, input_data, input_len=None, **cfg):
        if self.vad_model is None:
            return self.inference_with_asr(input_data, input_len=input_len, **cfg)

        else:
            return self.inference_with_vad(input_data, input_len=input_len, **cfg)

    @staticmethod
    def export_model(**kwargs):
        model, kwargs = AutoModel.build_model(**kwargs)

        if kwargs["compile_type"] == "punc":
            punc = MindiePunc(model)
            MindiePunc.export(punc, kwargs["compiled_path"], kwargs["soc_version"])
        elif kwargs["compile_type"] == "vad":
            vad = MindieVAD(model)
            MindieVAD.export(vad, kwargs["compiled_path"], kwargs["soc_version"])
        else:
            import copy
            from funasr.models.bicif_paraformer.export_meta import export_rebuild_model

            kwargs_new = copy.deepcopy(kwargs)
            kwargs_new['onnx'] = False
            kwargs_new["max_seq_len"] = 512
            del kwargs_new["model"]
            model = export_rebuild_model(model, **kwargs_new)

            encoder = MindieEncoder(model)
            MindieEncoder.export_ts(encoder, kwargs["compiled_encoder"], kwargs["soc_version"], kwargs["traced_encoder"])

            decoder = MindieDecoder(model)
            MindieDecoder.export_ts(decoder, kwargs["compiled_decoder"], kwargs["soc_version"], kwargs["traced_decoder"])
            
            mindie_cif = MindieCif(model.predictor.threshold, kwargs["cif_interval"])
            mindie_cif.export_ts(kwargs["compiled_cif"], kwargs["soc_version"])

            mindie_cif_timestamp = MindieCifTimestamp(model.predictor.threshold - 1e-4, kwargs["cif_timestamp_interval"])
            mindie_cif_timestamp.export_ts(kwargs["compiled_cif_timestamp"], kwargs["soc_version"])

    def build_model_with_mindie(self, **kwargs):
        assert "model" in kwargs
        if "model_conf" not in kwargs:
            logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
            kwargs = download_model(**kwargs)

        torch.set_num_threads(kwargs.get("ncpu", 4))

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["token_list"] = (
                tokenizer.token_list if hasattr(tokenizer, "token_list") else None
            )
            kwargs["token_list"] = (
                tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
            )
            vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
            if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = -1
        kwargs["tokenizer"] = tokenizer

        # build frontend
        frontend = kwargs.get("frontend", None)
        kwargs["input_size"] = None
        if frontend is not None:
            frontend_class = tables.frontend_classes.get(frontend)
            frontend = frontend_class(**kwargs.get("frontend_conf", {}))
            kwargs["input_size"] = (
                frontend.output_size() if hasattr(frontend, "output_size") else None
            )
        kwargs["frontend"] = frontend

        # build model
        model_conf = {}
        deep_update(model_conf, kwargs.get("model_conf", {}))
        deep_update(model_conf, kwargs)

        if kwargs["compile_type"] == "punc":
            model = MindieCTTransformer(**model_conf, vocab_size=vocab_size)
        else:
            model = MindieBiCifParaformer(**model_conf, vocab_size=vocab_size)

        # init_param
        init_param = kwargs.get("init_param", None)
        logging.info(f"Loading pretrained params from {init_param}")
        load_pretrained_model(
            model=model,
            path=init_param,
            ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
            oss_bucket=kwargs.get("oss_bucket", None),
            scope_map=kwargs.get("scope_map", []),
            excludes=kwargs.get("excludes", None),
        )

        return model, kwargs
    
    def inference_with_asr(self, input_data, input_len=None, model=None, kwargs=None, display_pbar=False, **cfg):
        kwargs = self.kwargs if kwargs is None else kwargs
        deep_update(kwargs, cfg)
        model = self.model if model is None else model
        model.eval()

        batch_size = kwargs.get("batch_size", 1)

        key_list, data_list = prepare_data_iterator(
            input_data, input_len=input_len, data_type=kwargs.get("data_type", None), key=None
        )

        time_stats = {"rtf_avg": 0.0, "input_speech_time": 0.0, "end_to_end_time": 0.0, "pure_infer_time": 0.0,
                      "load_data": 0.0, "encoder": 0.0, "predictor": 0.0, "decoder": 0.0,
                      "predictor_timestamp": 0.0, "post_process": 0.0}
        asr_result_list = []
        num_samples = len(data_list)

        if display_pbar:
            pbar = tqdm(colour="blue", total=num_samples, dynamic_ncols=True)

        for beg_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, beg_idx + batch_size)
            data_batch = data_list[beg_idx:end_idx]
            key_batch = key_list[beg_idx:end_idx]
            batch = {"data_in": data_batch, "key": key_batch}

            if (end_idx - beg_idx) == 1 and kwargs.get("data_type", None) == "fbank":  # fbank
                batch["data_in"] = data_batch[0]
                batch["data_lengths"] = input_len

            with torch.no_grad():
                time1 = time.perf_counter()
                res = model.inference_with_npu(**batch, **kwargs)
                time2 = time.perf_counter()
                if isinstance(res, (list, tuple)):
                    results = res[0] if len(res) > 0 else [{"text": ""}]
                    meta_data = res[1] if len(res) > 1 else {}

            asr_result_list.extend(results)

            batch_data_time = meta_data.get("batch_data_time", -1)
            time_escape = time2 - time1

            time_stats["load_data"] += meta_data.get("load_data", 0.0)
            time_stats["encoder"] += meta_data.get("encoder", 0.0)
            time_stats["predictor"] += meta_data.get("calc_predictor", 0.0)
            time_stats["decoder"] += meta_data.get("decoder", 0.0)
            time_stats["predictor_timestamp"] += meta_data.get("calc_predictor_timestamp", 0.0)
            time_stats["post_process"] += meta_data.get("post_process", 0.0)
            time_stats["end_to_end_time"] += time_escape

            time_stats["input_speech_time"] += batch_data_time

            time_stats["pure_infer_time"] = time_stats["end_to_end_time"] - time_stats["load_data"]
            time_stats["rtf_avg"] = time_stats["input_speech_time"] / time_stats["pure_infer_time"]

            if display_pbar:
                pbar.update(batch_size)
                pbar.set_description("rtf_avg:{:.3f}".format(time_stats["rtf_avg"]))

        return asr_result_list, time_stats