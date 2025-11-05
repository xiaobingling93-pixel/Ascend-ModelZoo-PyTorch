# MIT License
# Copyright (c) 2025 FunASR

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import argparse
import re
import time
from itertools import groupby
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

import jiwer
from datasets import load_dataset
from ais_bench.infer.interface import InferSession
from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.ctc.ctc import CTC
from funasr.models.sense_voice.utils.ctc_alignment import ctc_forced_align


class SenseVoiceOnnxModel():
    def __init__(self, device_id, om_path, model, vad_model):
        super().__init__()
        self.device = f"npu:{device_id}"
        _, self.kwargs = AutoModel.build_model(model=model, trust_remote_code=True)
        self.vad_model, self.vad_kwargs = AutoModel.build_model(model=vad_model, trust_remote_code=True)
        encoder_output_size = self.kwargs.get("encoder_conf", {}).get("output_size", 256)
        self.blank_id = 0
        self.vocab_size = self.kwargs.get("vocab_size", -1)
        self.frontend = self.kwargs.get("frontend", None)
        self.tokenizer = self.kwargs.get("tokenizer", None)
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.textnorm_dict = {'withitn': 14, "woitn": 15}
        self.om_sess = InferSession(device_id, om_path)
        self.ignore_id = -1
        ctc_conf = {}
        self.ctc = CTC(odim=self.vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)
        self.ctc.ctc_lo = self.ctc.ctc_lo.to(device=self.device)
    
    
    def is_valid_word(self, word):
        return word.isalpha() and word.isascii()

    # Copied from funasr.models.sense_voice.SenseVoiceEncoderSmall.post
    def post_process(self, timestamp):
        timestamp_new = []
        prev_word = None
        for i, t in enumerate(timestamp):
            word, start, end = t
            start = int(start * 1000)
            end = int(end * 1000)
            if word == "▁":
                continue
            if i == 0:
                timestamp_new.append([start, end, word])
            elif word.startswith("▁"):
                word = word[1:]
                timestamp_new.append([start, end, word])
            elif prev_word is not None and self.is_valid_word(prev_word) and self.is_valid_word(word):
                prev_word += word
                timestamp_new[-1][1] = end
                timestamp_new[-1][2] += word
            else:
                timestamp_new.append([start, end, word])
            prev_word = word
        return timestamp_new

    # Copied from funasr.models.sense_voice.SenseVoiceEncoderSmall.inference
    def sense_voice_infer(self, feed, vad_res_list, output_timestamp=False):
        custom_sizes = (feed[0].shape[1] + 4) * 4 * self.vocab_size # 根据输入shape预估输出占用显存大小
        ctc_logits, encoder_out_lens, encoder_out = self.om_sess.infer(feed, mode='dymshape', custom_sizes=custom_sizes)
        ctc_logits = torch.from_numpy(ctc_logits).to(device=self.device)
        encoder_out_lens = torch.from_numpy(encoder_out_lens).to(device=self.device)
        encoder_out = torch.from_numpy(encoder_out).to(device=self.device)
        x = ctc_logits[0, : encoder_out_lens[0].item(), :]
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)        
        mask = yseq != self.blank_id
        token_int = yseq[mask].tolist()
        text = self.tokenizer.decode(token_int)
        if not output_timestamp:
            return {'text': text}
        timestamp = []
        tokens = self.tokenizer.text2tokens(text)[4:]
        token_back_to_id = self.tokenizer.tokens2ids(tokens)
        token_ids = []
        for tok_ls in token_back_to_id:
            if tok_ls:
                token_ids.extend(tok_ls)
            else:
                token_ids.append(124)
        if len(token_ids) == 0:
            return {'text': text}
        logits_speech = self.ctc.softmax(encoder_out)[0, 4: encoder_out_lens[0].item(), :]
        pred = logits_speech.argmax(-1).cpu()
        logits_speech[pred == self.blank_id, self.blank_id] = 0
        align = ctc_forced_align(
            logits_speech.unsqueeze(0).float().cpu(),
            torch.Tensor(token_ids).unsqueeze(0).long(),
            (encoder_out_lens[0] - 4).long().cpu(),
            torch.tensor(len(token_ids)).unsqueeze(0).long(),
            ignore_id=self.ignore_id,
        )
        pred = groupby(align[0, : encoder_out_lens[0]])
        vad_offset = 30
        _start = (vad_res_list[0] + vad_offset) / 60
        token_id = 0
        ts_max = (vad_res_list[1] + vad_offset) / 60
        for pred_token, pred_frame in pred:
            _end = _start + len(list(pred_frame))
            if pred_token != 0:
                ts_left = max((_start * 60 - vad_offset) / 1000, 0)
                ts_right = min((_end * 60 - vad_offset) / 1000, (ts_max * 60 - vad_offset) / 1000)
                timestamp.append([tokens[token_id], ts_left, ts_right])
                token_id += 1
            _start = _end
        timestamp = self.post_process(timestamp)
        return {'text': text, 'timestamp': timestamp}


    def infer(self, data_in, output_timestamp):
        start_time = time.time()
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=self.frontend.fs,
            audio_fs=self.kwargs.get("fs", 16000),
            data_type=self.kwargs.get("data_type", "sound"),
            tokenizer=self.tokenizer
        )
        speech_list = []
        speech_lengths_list = []
        results, meta_data = self.vad_model.inference([data_in], key=['test'], **self.vad_kwargs)
        vad_res_list = results[0]['value']
        for start, end in vad_res_list:
            bed_idx = int(start * 16)
            end_idx = min(int(end * 16), len(audio_sample_list))
            sub_audio_sample_list = audio_sample_list[bed_idx:end_idx]
            speech, speech_lengths = extract_fbank(
                sub_audio_sample_list, data_type=self.kwargs.get("data_type", "sound"), frontend=self.frontend
            )
            speech = speech.to(device=self.device)
            speech_lengths = speech_lengths.to(device=self.device)
            speech_list.append(speech)
            speech_lengths_list.append(speech_lengths)
        language = self.kwargs.get("language", "auto")
        language = torch.LongTensor([self.lid_dict[language] if language in self.lid_dict else 0]).to(device=self.device)
        use_itn = self.kwargs.get('use_itn', True)
        textnorm = self.kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm = torch.LongTensor([self.textnorm_dict.get(textnorm, 0)]).to(device=self.device)
        results = {"key": data_in, "text": "", "timestamp": []}
        for speech, speech_lengths, vad_res in zip(speech_list, speech_lengths_list, vad_res_list):
            feed = [speech.cpu().detach().numpy().astype(np.float32),
                    speech_lengths.cpu().detach().numpy().astype(np.int32),
                    language.cpu().detach().numpy().astype(np.int32),
                    textnorm.cpu().detach().numpy().astype(np.int32)]
            result_i = self.sense_voice_infer(
                            feed=feed,
                            vad_res_list=vad_res,
                            output_timestamp=output_timestamp)
            results['text'] += result_i.get('text', '')
            results['timestamp'] += result_i.get('timestamp', [])
        e2e_time = time.time() - start_time
        print(f'infer E2E time = {e2e_time * 1000:.2f} ms')
        results['e2e_time'] = e2e_time
        return results


def process_output(text):
    text = re.sub(r'<\|.*?\|>', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.upper()
    return text


def benchmark(data_path, output_timestamp, model):
    hypotheses = []
    references = []
    total_e2e_time = 0
    total_audio_time = 0
    dataset = load_dataset("parquet", data_files=data_path).get('train', [])
    if not dataset or dataset[0].get('audio', {}).get('array', []) is None:
        print("benchmark fail, dataset is invalid")
        return
    #warmup
    res = model.infer(dataset[0]['audio']['array'], False)
    for data in dataset:
        array = data.get('audio', {}).get('array', [])
        if array is None:
            print("benchmark fail, dataset is invalid")
            return
        sampling_rate = data.get('audio', {}).get('sampling_rate', 16000)
        text = data.get('text', '')
        res = model.infer(array, output_timestamp)
        references.append(text)
        hypotheses.append(process_output(res['text']))
        total_e2e_time += res['e2e_time']
        total_audio_time += len(array) / sampling_rate
    print("Transcription Rate:", total_audio_time / total_e2e_time)
    wer = jiwer.wer(references, hypotheses)
    print("WER:", wer)


if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)
    parser = argparse.ArgumentParser(description="Sensevoice infer")
    parser.add_argument('--vad_path', type=str, help='vad path')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--om_path', type=str, help='om model')
    parser.add_argument('--device', type=int, help='device', default=0)
    parser.add_argument('--input', type=str, help='dataset path')
    parser.add_argument('--output_timestamp', action='store_true')
    args = parser.parse_args()
    model = SenseVoiceOnnxModel(args.device, args.om_path, args.model_path, args.vad_path)

    with torch.no_grad():
        benchmark(args.input, args.output_timestamp, model)