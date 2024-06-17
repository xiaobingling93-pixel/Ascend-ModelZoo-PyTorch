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

import argparse
import base64
import json
import math

import kaldi_native_fbank as knf
import numpy as np
import torch
import torchaudio
from ais_bench.infer.interface import InferSession
from tqdm import trange


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        default="encoder_linux_x86_64.om",
        help="Path to the encoder",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        default="decoder_linux_x86_64.om",
        help="Path to the decoder",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="tokens.txt",
        help="Path to the tokens",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="model_cfg.json",
        help="Path to the model config",
    )

    parser.add_argument(
        "--language",
        type=str,
        help="""The actual spoken language in the audio.
        Example values, en, de, zh, jp, fr.
        If None, we will detect the language using the first 30s of the
        input audio
        """,
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        type=str,
        default="transcribe",
        help="Valid values are: transcribe, translate",
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="Path to the test wave",
    )
    return parser.parse_args()


class OMModel:
    def __init__(
        self,
        cfg,
        encoder,
        decoder,
    ):
        self.encoder = InferSession(device_id=0, model_path=encoder)
        self.decoder = InferSession(device_id=0, model_path=decoder)
        self.load_cfg(cfg)

    def load_cfg(self, cfg):
        with open(cfg, "r") as json_file:
            model_cfg = json.load(json_file)
        self.n_text_layer = int(model_cfg["n_text_layer"])
        self.n_text_ctx = int(model_cfg["n_text_ctx"])
        self.n_text_state = int(model_cfg["n_text_state"])
        self.sot = int(model_cfg["sot"])
        self.eot = int(model_cfg["eot"])
        self.translate = int(model_cfg["translate"])
        self.transcribe = int(model_cfg["transcribe"])
        self.no_timestamps = int(model_cfg["no_timestamps"])
        self.no_speech = int(model_cfg["no_speech"])
        self.blank = int(model_cfg["blank_id"])
        self.vocab_size = int(model_cfg["n_vocab"])

        self.sot_sequence = list(map(int,
                                     model_cfg["sot_sequence"].split(",")))

        self.sot_sequence.append(self.no_timestamps)

        self.all_language_tokens = list(
            map(int, model_cfg["all_language_tokens"].split(",")))
        self.all_language_codes = model_cfg["all_language_codes"].split(",")
        self.lang2id = dict(
            zip(self.all_language_codes, self.all_language_tokens))
        self.id2lang = dict(
            zip(self.all_language_tokens, self.all_language_codes))

        self.is_multilingual = int(model_cfg["is_multilingual"]) == 1

    def get_self_cache(self):
        batch_size = 1
        n_layer_self_k_cache = np.zeros((self.n_text_layer, batch_size,
                                         self.n_text_ctx, self.n_text_state),
                                        dtype=np.float32)
        n_layer_self_v_cache = np.zeros((self.n_text_layer, batch_size,
                                         self.n_text_ctx, self.n_text_state),
                                        dtype=np.float32)
        return n_layer_self_k_cache, n_layer_self_v_cache

    def suppress_tokens(self, logits, is_initial: bool) -> None:
        # suppress blank
        if is_initial:
            logits[self.eot] = float("-inf")
            logits[self.blank] = float("-inf")

        # suppress <|notimestamps|>
        logits[self.no_timestamps] = float("-inf")

        logits[self.sot] = float("-inf")
        logits[self.no_speech] = float("-inf")

        # logits is changed in-place
        logits[self.translate] = float("-inf")

    def detect_language(self, n_layer_cross_k: torch.Tensor,
                        n_layer_cross_v: torch.Tensor) -> int:
        tokens = np.array([[self.sot]], dtype=np.int64)
        offset = np.zeros(1, dtype=np.int64)
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_self_cache()

        logits, n_layer_self_k_cache, n_layer_self_v_cache = self.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
        )
        logits = logits.reshape(-1)
        mask = np.ones(logits.shape[0], dtype=np.int64)
        mask[self.all_language_tokens] = 0
        logits[mask != 0] = float("-inf")
        lang_id = logits.argmax()
        print("detected language: ", self.id2lang[lang_id])
        return lang_id


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def compute_features(filename: str) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = torchaudio.load(filename)
    audio = wave[0].contiguous()  # only use the first channel
    if sample_rate != 16000:  #sample rate should be 16000, otherwise it will be resample
        audio = torchaudio.functional.resample(audio,
                                               orig_freq=sample_rate,
                                               new_freq=16000)

    features = []
    online_whisper_fbank = knf.OnlineWhisperFbank(knf.FrameExtractionOptions())
    online_whisper_fbank.accept_waveform(16000, audio.numpy())
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0

    # We pad 50 frames at the end so that it is able to detect eot
    # You can use another value instead of 50.
    mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)
    # Note that if it throws for a multilingual model,
    # please use a larger value, say 300

    target = 3000
    if mel.shape[0] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[:target - 50]
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)

    mel = mel.t().unsqueeze(0)

    return mel.numpy()


def main():
    args = get_args()

    mel = compute_features(args.sound_file)

    model = OMModel(args.model_cfg, args.encoder, args.decoder)
    n_layer_cross_k_bytesize = 6 * mel.shape[0] * math.ceil(
        mel.shape[2]) * 512 * 4
    n_layer_cross_v_bytesize = n_layer_cross_k_bytesize
    n_layer_cross_k, n_layer_cross_v = model.encoder.infer(
        [
            mel,
        ],
        mode='dymshape',
        custom_sizes=[n_layer_cross_k_bytesize, n_layer_cross_v_bytesize])
    
    if args.language is not None:
        if model.is_multilingual is False and args.language != "en":
            print(f"This model supports only English. Given: {args.language}")
            return

        if args.language not in model.lang2id:
            print(f"Invalid language: {args.language}")
            print(f"Valid values are: {list(model.lang2id.keys())}")
            return

        # [sot, lang, task, notimestamps]
        model.sot_sequence[1] = model.lang2id[args.language]
    elif model.is_multilingual is True:
        print("detecting language")
        lang = model.detect_language(n_layer_cross_k, n_layer_cross_v)
        model.sot_sequence[1] = lang

    if args.task is not None:
        if model.is_multilingual is False and args.task != "transcribe":
            print(
                "This model supports only English. Please use --task=transcribe"
            )
            return
        assert args.task in ["transcribe", "translate"], args.task

        if args.task == "translate":
            model.sot_sequence[2] = model.translate

    n_layer_self_k_cache, n_layer_self_v_cache = model.get_self_cache()
    tokens = np.array([model.sot_sequence], dtype=np.int64)
    offset = np.zeros(1, dtype=np.int64)

    logits_bytesize = mel.shape[0] * 2 * model.vocab_size * 4
    n_layer_self_k_cache_bytesize = 6 * mel.shape[0] * 448 * 512 * 4
    n_layer_self_v_cache_bytesize = n_layer_self_k_cache_bytesize
    logits, n_layer_self_k_cache, n_layer_self_v_cache = model.decoder.infer(
        [
            tokens, n_layer_self_k_cache, n_layer_self_v_cache,
            n_layer_cross_k, n_layer_cross_v, offset
        ],
        mode="dymshape",
        custom_sizes=[
            logits_bytesize, n_layer_self_k_cache_bytesize,
            n_layer_self_v_cache_bytesize
        ])

    offset += len(model.sot_sequence)

    logits = logits[0, -1]
    model.suppress_tokens(logits, is_initial=True)
    max_token_id = logits.argmax(axis=-1)
    results = []
    for i in trange(model.n_text_ctx):
        if max_token_id == model.eot:
            break
        results.append(max_token_id)

        tokens = np.array([[results[-1]]])

        logits, n_layer_self_k_cache, n_layer_self_v_cache = model.decoder.infer(
            [
                tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                n_layer_cross_k, n_layer_cross_v, offset
            ],
            mode="dymshape",
            custom_sizes=[
                logits_bytesize, n_layer_self_k_cache_bytesize,
                n_layer_self_v_cache_bytesize
            ])
        offset += 1
        logits = logits[0, -1]
        model.suppress_tokens(logits, is_initial=False)
        max_token_id = logits.argmax(axis=-1)
    token_table = load_tokens(args.tokens)
    s = b""
    for i in results:
        if i in token_table:
            s += base64.b64decode(token_table[i])

    print(s.decode().strip())


if __name__ == "__main__":
    main()
