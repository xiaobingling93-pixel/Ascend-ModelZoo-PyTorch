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
import subprocess
from functools import lru_cache
from typing import Union, Optional

import torch
import torch.nn.functional as F

import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

import numpy as np
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

import whisper
from funasr import AutoModel
from funasr.utils.vad_utils import merge_vad
import librosa

from modeling_whisper import get_whisper_model


def exact_div(x1, x2):
    if x2 == 0:
        raise ValueError("x2 cannot be zero (division by zero is not allowed)")
    if x1 % x2 != 0:
        raise ValueError("x1 is not divisible by x2")
    return x1 // x2


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    mel_filters_path = os.path.join(os.path.dirname(__file__), "mel_filters.npz")
    if not os.path.exists(mel_filters_path):
        np.savez_compressed(
            mel_filters_path,
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    if n_mels not in [80, 128]:
        raise ValueError(f"Unsupported n_mels: {n_mels}")
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
    

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    else:
        audio = audio[..., :N_SAMPLES]
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    stft_real_img = torch.view_as_real(stft)[..., :-1, :]
    real_part = stft_real_img[..., 0]
    img_part = stft_real_img[..., 1]
    magnitudes = real_part ** 2 + img_part ** 2

    filters = mel_filters(device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if log_spec.shape[-1] != 3000:
        raise ValueError("log_spec shape is not as expected")
    return log_spec


class TorchairPipeline(Pipeline):
    def __init__(self, whisper_model_path, vad_model_path, batch_size, device_id, whisper_decode_options, **kwargs):
        self.device = torch.device('npu:{}'.format(device_id))
        self.whisper_decode_options = whisper_decode_options

        print("start load whisper")
        self.model = get_whisper_model(whisper_model_path, whisper_decode_options, self.device)
        
        print("start load vad")
        self.vad_model = AutoModel(model=vad_model_path, disable_update=True)
        self.vad_model.model = self.vad_model.model.to(self.device)

        torch_npu.npu.set_compile_mode(jit_compile=False)
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True  # 使能tiling全下沉配置
        npu_backend = tng.get_npu_backend(compiler_config=config)
        
        print("compile model...")
        self.model.encoder.forward = torch.compile(self.model.encoder.forward, dynamic=False, fullgraph=True, backend=npu_backend)
        self.model.prefill_decoder.forward = torch.compile(self.model.prefill_decoder.forward, dynamic=False, fullgraph=True, backend=npu_backend)
        self.model.decode_decoder.forward = torch.compile(self.model.decode_decoder.forward, dynamic=True, fullgraph=True, backend=npu_backend)
        self.vad_model.model = torch.compile(self.vad_model.model, dynamic=True, fullgraph=True, backend=npu_backend)

        self._batch_size = batch_size
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = "pt"

        super(Pipeline, self).__init__()

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}
    
    def preprocess(self, audio):
        model_n_mels = self.model.dims.n_mels
        audio = audio["inputs"]
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - len(audio),
            device=self.device
        )
        return {"inputs": features}
    
    def _forward(self, model_inputs, **generate_kwargs):
        audio_mel = model_inputs["inputs"]
        result = whisper.decode(self.model, audio_mel, self.whisper_decode_options)
        text = [res.text for res in result]
        language = [res.language for res in result]
        return {"text": text, "language": language}
    
    def forward(self, model_inputs, **forward_params):
        model_outputs = self._forward(model_inputs, **forward_params)
        return model_outputs

    def postprocess(self, model_outputs):
        return model_outputs
    
    def get_iterator(self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)

        def stack(items):
            return {"inputs": torch.stack([x["inputs"] for x in items])}
        
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)

        return final_iterator
    
    def pad_segment(self, vad_segments, batch_size):
        need_pad_num = batch_size - len(vad_segments) % batch_size
        paded_segments = vad_segments
        for _ in range(need_pad_num):
            paded_segments.append(vad_segments[-1])
        return paded_segments
    
    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, chunk_size=30
    ):
        segments = []

        def data(total_audio, segments):
            for seg in segments:
                f1 = int(seg[0] * SAMPLE_RATE / 1000) # funasr vad count time in ms
                f2 = int(seg[1] * SAMPLE_RATE / 1000)
                yield {"inputs": total_audio[f1:f2]}
        
        vad_segments = self.vad_model.generate(torch.from_numpy(audio).to(self.device))
        vad_segments = merge_vad(vad_segments[0]["value"], chunk_size * 1000)
        total_segments = len(vad_segments)

        if total_segments % batch_size != 0:
            vad_segments = self.pad_segment(vad_segments, batch_size)

        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            text, language = out["text"], out["language"]
            if batch_size in [0, 1, None]:
                text = text[0]
                language = language[0]
            segments.append(
                {
                    "language": language,
                    "text": text,
                    "start": round(vad_segments[idx][0] / 1000, 3),
                    "end": round(vad_segments[idx][1] / 1000, 3)
                }
            )
        return segments