# Copyright 2024 Huawei Technologies Co., Ltd
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

#!/usr/bin/python
#encoding=utf-8

import os
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import whisper
from torch import Tensor, nn
from whisper.model import (
    AudioEncoder,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="./base.en.pt")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./whisper_base_en")
    return parser.parse_args()


def modified_audio_encoder_forward(self: AudioEncoder, x: torch.Tensor):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    assert (
        x.shape[2] == self.positional_embedding.shape[1]
    ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
    assert (
        x.shape[1] == self.positional_embedding.shape[0]
    ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
    x = (x + self.positional_embedding[:x.shape[1]]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x


AudioEncoder.forward = modified_audio_encoder_forward


class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder,
                 inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor):
        audio_features = self.audioEncoder(x)

        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for block in self.textDecoder.blocks:
            n_layer_cross_k_list.append(block.cross_attn.key(audio_features))
            n_layer_cross_v_list.append(block.cross_attn.value(audio_features))

        return torch.stack(n_layer_cross_k_list), torch.stack(
            n_layer_cross_v_list)


class MultiHeadAttentionCross(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv)


class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,  # (b, n_ctx      , n_state)
        k_cache: Tensor,  # (b, n_ctx_cache, n_state)
        v_cache: Tensor,  # (b, n_ctx_cache, n_state)
        mask: Tensor,
    ):
        q = self.multiHeadAttention.query(x)  # (b, n_ctx, n_state)
        k = self.multiHeadAttention.key(x)  # (b, n_ctx, n_state)
        v = self.multiHeadAttention.value(x)  # (b, n_ctx, n_state)

        k_cache[:, -k.shape[1]:, :] = k  # (b, n_ctx_cache + n_ctx, n_state)
        v_cache[:, -v.shape[1]:, :] = v  # (b, n_ctx_cache + n_ctx, n_state)

        wv, qk = self.multiHeadAttention.qkv_attention(q, k_cache, v_cache,
                                                       mask)
        return self.multiHeadAttention.out(wv), k_cache, v_cache


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, inResidualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelf(inResidualAttentionBlock.attn)
        self.cross_attn = (MultiHeadAttentionCross(
            inResidualAttentionBlock.cross_attn)
                           if inResidualAttentionBlock.cross_attn else None)

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        mask: Tensor,
    ):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(
            self.originalBlock.attn_ln(x),
            self_k_cache,
            self_v_cache,
            mask=mask)
        x = x + self_attn_x

        if self.cross_attn:
            x = x + self.cross_attn(self.originalBlock.cross_attn_ln(x),
                                    cross_k, cross_v)

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k_cache_updated, self_v_cache_updated


class TextDecoderTensorCache(nn.Module):
    def __init__(self, inTextDecoder: TextDecoder, in_n_ctx: int):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(
                ResidualAttentionBlockTensorCache(orginal_block))

    def forward(
        self,
        tokens: Tensor,
        n_layer_self_k_cache: Tensor,
        n_layer_self_v_cache: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
        offset: Tensor,
    ):
        x = (self.textDecoder.token_embedding(tokens) +
             self.textDecoder.positional_embedding[offset[0]:offset[0] +
                                                   tokens.shape[-1]])
        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[i, :, :offset[0] +
                                                tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[i, :, :offset[0] +
                                                tokens.shape[-1], :]
            x, self_k_cache, self_v_cache = block(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[i],
                cross_v=n_layer_cross_v[i],
                mask=self.textDecoder.mask,
            )
            n_layer_self_k_cache[i, :, :offset[0] +
                                 tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[i, :, :offset[0] +
                                 tokens.shape[-1], :] = self_v_cache
            i += 1

        x = self.textDecoder.ln(x)
        
        logits = (torch.matmul(
            self.textDecoder.token_embedding.weight.to(x.dtype),
            x.permute(0, 2, 1),
        ).permute(0, 2, 1).float())

        return logits, n_layer_self_k_cache, n_layer_self_v_cache


def convert_tokens(model, output_path):
    whisper_dir = Path(whisper.__file__).parent
    multilingual = model.is_multilingual
    tokenizer = (whisper_dir / "assets" /
                 (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken"))
    if not tokenizer.is_file():
        raise ValueError(f"Cannot find {tokenizer}")

    tokens = {}
    with open(tokenizer, "r") as f:
        contents = f.read()
        for line in contents.splitlines():
            if not line:
                continue
            token, rank = line.split()
            tokens[token] = int(rank)

    with open(f"{output_path}/tokens.txt", "w") as f:
        for t, i in tokens.items():
            f.write(f"{t} {i}\n")


def get_cfg(model, output_path):
    convert_tokens(model, output_path)
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    model_cfg = {
        "n_mels":
        model.dims.n_mels,
        "n_audio_ctx":
        model.dims.n_audio_ctx,
        "n_audio_state":
        model.dims.n_audio_state,
        "n_audio_head":
        model.dims.n_audio_head,
        "n_audio_layer":
        model.dims.n_audio_layer,
        "n_vocab":
        model.dims.n_vocab,
        "n_text_ctx":
        model.dims.n_text_ctx,
        "n_text_state":
        model.dims.n_text_state,
        "n_text_head":
        model.dims.n_text_head,
        "n_text_layer":
        model.dims.n_text_layer,
        "sot_sequence":
        ",".join(list(map(str, tokenizer.sot_sequence))),
        "all_language_tokens":
        ",".join(list(map(str,
                          tokenizer.all_language_tokens))),  # a list of ids
        "all_language_codes":
        ",".join(tokenizer.all_language_codes),  # e.g., en, de, zh, fr
        "sot":
        tokenizer.sot,
        "sot_index":
        tokenizer.sot_sequence.index(tokenizer.sot),
        "eot":
        tokenizer.eot,
        "blank_id":
        tokenizer.encode(" ")[0],
        "is_multilingual":
        int(model.is_multilingual),
        "no_speech":
        tokenizer.no_speech,
        "non_speech_tokens":
        ",".join(list(map(str, tokenizer.non_speech_tokens))),
        "transcribe":
        tokenizer.transcribe,
        "translate":
        tokenizer.translate,
        "sot_prev":
        tokenizer.sot_prev,
        "sot_lm":
        tokenizer.sot_lm,
        "no_timestamps":
        tokenizer.no_timestamps,
    }
    with open(f"{output_path}/model_cfg.json", "w") as json_file:
        json.dump(model_cfg, json_file)


@torch.no_grad()
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    opset_version = 14
    model = whisper.load_model(args.model)

    print(
        "number of model parameters:",
        sum(p.numel() for p in model.parameters()),
    )
    print(
        "number of encoder parameters:",
        sum(p.numel() for p in model.encoder.parameters()),
    )
    print(
        "number of decoder parameters:",
        sum(p.numel() for p in model.decoder.parameters()),
    )

    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

    model.eval()
    get_cfg(model, args.output_dir)
    audio = torch.rand(16000 * 2)
    audio = whisper.pad_or_trim(audio)
    assert audio.shape == (16000 * 30, ), audio.shape

    # make log-Mel spectrogram and move to the same device as the model
    n_mels = model.dims.n_mels
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(model.device).unsqueeze(0)
    batch_size = 1
    assert mel.shape == (batch_size, n_mels, 30 * 100)

    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)

    n_layer_cross_k, n_layer_cross_v = encoder(mel)
    assert n_layer_cross_k.shape == (
        model.dims.n_text_layer,
        batch_size,
        model.dims.n_audio_ctx,
        model.dims.n_text_state,
    ), (n_layer_cross_k.shape, model.dims)
    assert n_layer_cross_v.shape == (
        model.dims.n_text_layer,
        batch_size,
        model.dims.n_audio_ctx,
        model.dims.n_text_state,
    ), (n_layer_cross_v.shape, model.dims)

    encoder_filename = f"{args.output_dir}/encoder.onnx"
    torch.onnx.export(
        encoder,
        mel,
        encoder_filename,
        opset_version=opset_version,
        input_names=["mel"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
        dynamic_axes={
            "mel": {
                0: "n_audio",
                2: "T"
            },  # n_audio is also known as batch_size
            "n_layer_cross_k": {
                1: "n_audio",
                2: "T"
            },
            "n_layer_cross_v": {
                1: "n_audio",
                2: "T"
            },
        },
    )

    n_audio = mel.shape[0]
    tokens = torch.tensor([[tokenizer.sot, tokenizer.sot, tokenizer.sot]] *
                          n_audio).to(mel.device)  # [n_audio, 3]
    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)
    n_layer_self_k_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    offset = torch.zeros(1, dtype=torch.int64).to(mel.device)
    logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        offset,
    )
    assert logits.shape == (n_audio, tokens.shape[1], model.dims.n_vocab)
    assert n_layer_self_k_cache.shape == (
        model.dims.n_text_layer,
        n_audio,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
    )
    assert n_layer_self_v_cache.shape == (
        model.dims.n_text_layer,
        n_audio,
        model.dims.n_text_ctx,
        model.dims.n_text_state,
    )

    offset = torch.tensor([tokens.shape[1]], dtype=torch.int64).to(mel.device)
    tokens = torch.tensor([[tokenizer.sot]] * n_audio).to(
        mel.device)  # [n_audio, 1]

    logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = decoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        offset,
    )

    decoder_filename = f"{args.output_dir}/decoder.onnx"
    torch.onnx.export(
        decoder,
        (
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        ),
        decoder_filename,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        ],
        output_names=[
            "logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"
        ],
        dynamic_axes={
            "tokens": {
                0: "n_audio",
                1: "n_tokens"
            },
            "in_n_layer_self_k_cache": {
                1: "n_audio"
            },
            "in_n_layer_self_v_cache": {
                1: "n_audio"
            },
            "n_layer_cross_k": {
                1: "n_audio",
                2: "T"
            },
            "n_layer_cross_v": {
                1: "n_audio",
                2: "T"
            },
        },
    )


if __name__ == "__main__":
    main()
