# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import time
import math
import argparse
from typing import Optional

import jiwer
import numpy as np
import pandas as pd
from datasets import load_dataset
import librosa

import torch
from torch import nn, Tensor
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

import whisper
from whisper.model import Linear
from whisper.decoding import PyTorchInference, DecodingResult, DecodingTask
from whisper.normalizers import EnglishTextNormalizer

from rewrited_models import PrefillTextDecoder, DecodeTextDecoder


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, speech_path, device, audio_column="audio", text_column='text'):
        self.dataset = load_dataset(speech_path, split="validation")
        self.audio_column = audio_column
        self.text_column = text_column
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 自动解码音频 + 重采样到 16kHz
        audio = self.dataset[idx]["audio"]["array"]  # 直接获取 NumPy 数组
        audio = torch.from_numpy(audio).float()

        # 统一长度 + 生成梅尔频谱
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        return mel.contiguous().to(self.device), self.dataset[idx][self.text_column]


def parse_args():
    parser = argparse.ArgumentParser("Whisper infer")
    parser.add_argument("--model_path", type=str, default="./base.pt", help="model checkpoint file path")
    parser.add_argument("--audio_path", type=str, default="./audio.mp3",
                        help="warmup audio file path")
    parser.add_argument("--speech_path", type=str, default="./librispeech_asr_dummy/clean/",
                        help="librispeech_asr_dummy english transaction speech data path")
    parser.add_argument('--device', type=int, default='0', help="npu device id")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--warmup', type=int, default=5, help="Warm up times")
    parser.add_argument('--loop', type=int, default=5, help="Loop times")
    args = parser.parse_args()
    return args


def create_model(args):
    model = whisper.load_model(args.model_path)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    return model


def rewrite_multi_head_attention_forward(model):
    wk = model.key.weight
    wv = model.value.weight
    model.kv = Linear(in_features=wk.shape[0], out_features=wk.shape[1] + wv.shape[1])
    model.kv.weight = nn.Parameter(torch.concat([wk, wv], dim=0), requires_grad=False)
    wk_bias = model.key.bias if model.key.bias is not None else torch.zeros(wk.shape[0])
    wv_bias = model.value.bias if model.value.bias is not None else torch.zeros(wv.shape[0])
    model.kv.bias = nn.Parameter(torch.concat([wk_bias, wv_bias], dim=0), requires_grad=False)

    def forward(
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            kv_cache: Optional[dict] = None,
            actual_seq_len: Optional[list] = None,
    ):
        q = model.query(x)

        # encoder
        if kv_cache is None:
            kv = model.kv(x)
            k, v = kv.chunk(2, dim=-1)

        # decoder - cross_attention
        if kv_cache is not None and xa is not None:
            k_key = "key"
            v_key = "value"
            if k_key in kv_cache:
                k = kv_cache[k_key]
                v = kv_cache[v_key]
            else:
                kv = model.kv(xa)
                k, v = kv.chunk(2, dim=-1)
                kv_cache[k_key] = k.contiguous()
                kv_cache[v_key] = v.contiguous()

        # decoder - self_attention
        if kv_cache is not None and xa is None:
            k_key = "key"
            v_key = "value"
            if k_key in kv_cache:
                k = kv_cache[k_key]
                v = kv_cache[v_key]
                new_kv = model.kv(x[:, -1:])
                new_k = new_kv[..., :wk.shape[0]]
                new_v = new_kv[..., wk.shape[0]:]
                kv_cache[k_key] = torch.cat([k.contiguous(), new_k.contiguous()], dim=1).detach()
                kv_cache[v_key] = torch.cat([v.contiguous(), new_v.contiguous()], dim=1).detach()
                k, v = kv_cache[k_key], kv_cache[v_key]
            else:
                kv = model.kv(x)
                k, v = kv.chunk(2, dim=-1)
                kv_cache[k_key] = k.contiguous()
                kv_cache[v_key] = v.contiguous()

        n_batch, n_ctx, n_state = q.shape
        q = q.view(*q.shape[:2], model.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], model.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], model.n_head, -1).permute(0, 2, 1, 3)

        mask = mask.to(torch.bool) if mask is not None and n_ctx > 1 else None
        sparse_mode = 1 if mask is not None and n_ctx > 1 else 0
        D = n_state // model.n_head

        at = torch_npu.npu_prompt_flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            num_heads=model.n_head,
            input_layout="BNSD",
            scale_value=1 / math.sqrt(D),
            atten_mask=mask[:n_ctx, :n_ctx] if mask is not None else None,
            sparse_mode=sparse_mode
        )

        qk = None
        w_v = at.permute(0, 2, 1, 3).flatten(start_dim=2)
        return model.out(w_v), qk

    model.forward = forward


def modify_model(model, options, args, device):
    print("modify model...")

    # 修改encoder的attention forward
    for block1, block2 in zip(model.encoder.blocks, model.decoder.blocks):
        rewrite_multi_head_attention_forward(block1.attn)
        rewrite_multi_head_attention_forward(block2.attn)
        rewrite_multi_head_attention_forward(block2.cross_attn)
    origin_decoder = model.decoder

    # 将原本的decoder拆分成prefill和decode2个阶段
    prefill_decoder = PrefillTextDecoder(
        model.dims.n_vocab,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
        model.dims.n_text_head,
        model.dims.n_text_layer
    )
    prefill_decoder.load_state_dict(origin_decoder.state_dict())

    decode_decoder = DecodeTextDecoder(
        model.dims.n_vocab,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
        model.dims.n_text_head,
        model.dims.n_text_layer
    )
    decode_decoder.load_state_dict(origin_decoder.state_dict())

    model.prefill_decoder = prefill_decoder
    model.decode_decoder = decode_decoder

    if options.fp16:
        model = model.half()
        for module in model.modules():
            # 在Whisper源码中，LayerNorm层需要接收fp32数据，因此需要特殊处理
            if isinstance(module, nn.LayerNorm):
                module = module.float()

    return model.eval().to(device)


def rewrite_inference_logits():
    def _patched_logits(self, tokens, audio_features) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
            self.kv_cache = [
                {
                    'attn': {},
                    'cross_attn': {}
                }
                for _ in range(6)
            ]
            return self.model.prefill_decoder(tokens, audio_features, kv_cache=self.kv_cache)

        actual_seq_len = tokens.shape[-1]
        updated_kv_positions = torch.tensor([actual_seq_len - 1], dtype=torch.long, device=tokens.device)
        kv_padding_size = torch.tensor([448 - actual_seq_len], dtype=torch.long, device=tokens.device)

        offset = actual_seq_len - 1
        positional_embedding = self.model.decode_decoder.positional_embedding[offset: offset + 1]
        tokens = tokens[:, -1:].contiguous().clone()

        torch._dynamo.mark_static(tokens)
        torch._dynamo.mark_static(audio_features)
        torch._dynamo.mark_static(positional_embedding)
        for i in range(6):
            torch._dynamo.mark_static(self.kv_cache[i]['attn']["key"])
            torch._dynamo.mark_static(self.kv_cache[i]['attn']["value"])
            torch._dynamo.mark_static(self.kv_cache[i]['cross_attn']["key"])
            torch._dynamo.mark_static(self.kv_cache[i]['cross_attn']["value"])
        torch._dynamo.mark_static(kv_padding_size)

        return self.model.decode_decoder(tokens, audio_features, positional_embedding, self.kv_cache,
                                         actual_seq_len=[actual_seq_len], kv_padding_size=kv_padding_size,
                                         updated_kv_positions=updated_kv_positions)

    PyTorchInference.logits = _patched_logits


def model_compile():
    print("torch.compile...")
    wsp_model.encoder.forward = torch.compile(wsp_model.encoder.forward, dynamic=False, fullgraph=True, backend=npu_backend)
    wsp_model.prefill_decoder.forward = torch.compile(wsp_model.prefill_decoder.forward, dynamic=False, fullgraph=True, backend=npu_backend)
    wsp_model.decode_decoder.forward = torch.compile(wsp_model.decode_decoder.forward, dynamic=True, fullgraph=True, backend=npu_backend)


def libri_speech_infer(model, options, loader):
    hypotheses = []
    references = []
    e2e_time_list = []

    for mels, texts in loader:
        start_time = time.time()
        results = model.decode(mels, options)
        e2e_time = time.time() - start_time
        e2e_time_list.append(e2e_time)
        print(f'Parquet infer E2E time = {e2e_time * 1000:.2f} ms')
        hypotheses.extend([res.text for res in results])
        references.extend(texts)

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    avg_e2e_time = sum(e2e_time_list) / len(e2e_time_list)
    print(data)
    normalizer = EnglishTextNormalizer()
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    print(data[["hypothesis_clean", "reference_clean"]])
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    return wer, avg_e2e_time


if __name__ == '__main__':
    wsp_args = parse_args()
    device = torch.device('npu:{}'.format(wsp_args.device))

    torch_npu.npu.set_compile_mode(jit_compile=False)
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True  # 使能tiling全下沉配置
    npu_backend = tng.get_npu_backend(compiler_config=config)

    dataset = LibriSpeechDataset(wsp_args.speech_path, device=device)
    audios = load_dataset(wsp_args.speech_path, split="validation")
    duration_seconds = 0
    for audio in audios:
        y, audio_sr = audio["audio"]["array"], audio["audio"]["sampling_rate"]
        duration_seconds += librosa.get_duration(y=y, sr=audio_sr)

    loader = torch.utils.data.DataLoader(dataset, batch_size=wsp_args.batch_size)
    options = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=True)

    wsp_model = create_model(wsp_args)
    wsp_model = modify_model(wsp_model, options, wsp_args, device)

    rewrite_inference_logits()
    model_compile()

    with torch.inference_mode():
        audio = whisper.load_audio(wsp_args.audio_path)
        audio = whisper.pad_or_trim(audio)
        audio_mel = whisper.log_mel_spectrogram(audio, n_mels=wsp_model.dims.n_mels).to(wsp_model.device)
        audio_mel = audio_mel.unsqueeze(0).repeat(wsp_args.batch_size, 1, 1)
        w_options = whisper.DecodingOptions(language='zh', without_timestamps=True, fp16=True)
        for _step in range(wsp_args.warmup):
            result = whisper.decode(wsp_model, audio_mel, w_options)
            for bs in range(wsp_args.batch_size):
                print("{}/{} - {}".format(_step, wsp_args.warmup, result[bs].text))

        print("LibriSpeech infer, English to English TRANSCRIBE ...")
        e2e_time_list = []
        for _ in range(wsp_args.loop):
            p_wer, e2e_time = libri_speech_infer(wsp_model, options, loader)
            e2e_time_list.append(e2e_time)
        print(f"LibriSpeech infer WER score =  {p_wer * 100:.2f} %")
        print(f"Average E2E infer time = {(sum(e2e_time_list)/wsp_args.loop)*1000:.2f}ms")

