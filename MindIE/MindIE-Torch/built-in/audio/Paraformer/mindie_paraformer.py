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

import copy
import time
import torch
import torch.nn.functional as F

from funasr.models.bicif_paraformer.model import BiCifParaformer, load_audio_text_image_video, \
    extract_fbank, Hypothesis, ts_prediction_lfr6_standard, postprocess_utils


COSINE_THRESHOLD = 0.999


def cosine_similarity(gt_tensor, pred_tensor):
    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res = torch.nn.functional.cosine_similarity(gt_tensor, pred_tensor, dim=0, eps=1e-6)
    res = res.cpu().detach().item()
    return res


def precision_eval(mrt_res, ref_res):
    if not isinstance(mrt_res, (list, tuple)):
        mrt_res = [mrt_res, ]
    if not isinstance(ref_res, (list, tuple)):
        ref_res = [ref_res, ]

    com_res = True
    for j, a in zip(mrt_res, ref_res):
        res = cosine_similarity(j.to("cpu"), a)
        print(res)
        if res < COSINE_THRESHOLD:
            com_res = False

    if com_res:
        print("Compare success! NPU model have the same output with CPU model!")
    else:
        print("Compare failed! Outputs of NPU model are not the same with CPU model!")


class MindieBiCifParaformer(BiCifParaformer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mindie_encoder = torch.jit.load(kwargs["compiled_encoder"])
        self.mindie_decoder = torch.jit.load(kwargs["compiled_decoder"])
        self.mindie_cif = torch.jit.load(kwargs["compiled_cif"])
        self.mindie_cif_timestamp = torch.jit.load(kwargs["compiled_cif_timestamp"])

    def inference_with_npu(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        # Step1: load input data
        time1 = time.perf_counter()
        meta_data = {}

        is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
        is_use_lm = (
            kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
        )
        if self.beam_search is None and (is_use_lm or is_use_ctc):
            self.init_beam_search(**kwargs)     
            self.nbest = kwargs.get("nbest", 1)
        audio_sample_list = load_audio_text_image_video(
            data_in, fs=frontend.fs, audio_fs=kwargs.get("fs", 16000)
        )

        speech, speech_lengths = extract_fbank(
            audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
        )
        meta_data["batch_data_time"] = (
            speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
        )
        speech = speech.to("npu")
        speech_lengths = speech_lengths.to("npu")

        time2 = time.perf_counter()
        meta_data["load_data"] = time2 - time1

        # Step2: run with compiled encoder
        encoder_out, hidden, alphas, pre_token_length = self.mindie_encoder(speech, speech_lengths)
        encoder_out_lens = speech_lengths

        hidden = hidden.to(kwargs["mindie_device"])
        alphas = alphas.to(kwargs["mindie_device"])
        pre_token_length = pre_token_length.to(kwargs["mindie_device"])

        pre_token_length = pre_token_length.round().to(torch.int32)
        time3 = time.perf_counter()
        meta_data["encoder"] = time3 - time2


        # Step3: divide dynamic loop into multiple smaller loops for calculation
        # each with a number of iterations based on kwargs["cif_interval"]
        batch_size, len_time, hidden_size = hidden.size()
        loop_num = len_time // kwargs["cif_interval"] + 1
        padding_len = loop_num * kwargs["cif_interval"]
        padding_size = padding_len - len_time
        padded_hidden = F.pad(hidden, (0, 0, 0, padding_size), "constant", 0)
        padded_alphas = F.pad(alphas, (0, padding_size), "constant", 0)

        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()

        frames_batch = []
        for b in range(batch_size):
            frames_list = []
            integrate = torch.zeros([1, ]).to("npu")
            frame = torch.zeros([1, hidden_size]).to("npu")
            for i in range(loop_num):
                cur_hidden = padded_hidden[b : b + 1, i * kwargs["cif_interval"] : (i + 1) * kwargs["cif_interval"], :]
                cur_alphas = padded_alphas[b : b + 1, i * kwargs["cif_interval"] : (i + 1) * kwargs["cif_interval"]]
                cur_frames, integrate, frame = self.mindie_cif(cur_hidden.to("npu"), cur_alphas.to("npu"), integrate, frame)
                frames_list.append(cur_frames.to(kwargs["mindie_device"]))
            frame = torch.cat(frames_list, 0)
            pad_frame = torch.zeros([max_label_len - frame.size(0), hidden_size], device=hidden.device)
            frames_batch.append(torch.cat([frame, pad_frame], 0))

        acoustic_embeds = torch.stack(frames_batch, 0)
        token_num_int = torch.max(pre_token_length)
        pre_acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        if torch.max(pre_token_length) < 1:
            return []
        time4 = time.perf_counter()
        meta_data["calc_predictor"] = time4 - time3


        # Step4: run with compiled decoder
        decoder_out, us_alphas = self.mindie_decoder(encoder_out, encoder_out_lens,
                                                     pre_acoustic_embeds.contiguous().to("npu"), pre_token_length.contiguous().to("npu"))
        us_alphas = us_alphas.to(kwargs["mindie_device"])
        time5 = time.perf_counter()
        meta_data["decoder"] = time5 - time4


        # Step5: divide dynamic loop into multiple smaller loops for calculation
        # each with a number of iterations based on kwargs["cif_timestamp_interval"]
        len_alphas = us_alphas.shape[1]
        loop_num = len_alphas // kwargs["cif_timestamp_interval"] + 1
        padding_len = loop_num * kwargs["cif_timestamp_interval"]
        padding_size = padding_len - len_alphas
        padded_alphas = F.pad(us_alphas, (0, padding_size), "constant", 0)

        peak_batch = []
        for b in range(batch_size):
            peak_list = []
            integrate_alphas = torch.zeros([1]).to("npu")
            for i in range(loop_num):
                cur_alphas = padded_alphas[b : b + 1, i * kwargs["cif_timestamp_interval"] : (i + 1) * kwargs["cif_timestamp_interval"]]
                peak, integrate_alphas = self.mindie_cif_timestamp(cur_alphas.to("npu"), integrate_alphas)
                peak_list.append(peak.to(kwargs["mindie_device"]))
            us_peak = torch.cat(peak_list, 1)[:, :len_alphas]
            peak_batch.append(us_peak)
        us_peaks = torch.cat(peak_batch, 0)

        time6 = time.perf_counter()
        meta_data["calc_predictor_timestamp"] = time6 - time5


        # Step6: post process
        decoder_out = decoder_out.to(kwargs["mindie_device"])
        us_alphas = us_alphas.to(kwargs["mindie_device"])
        us_peaks = us_peaks.to(kwargs["mindie_device"])
        encoder_out_lens = encoder_out_lens.to(kwargs["mindie_device"])
        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            am_scores = decoder_out[i, : pre_token_length[i], :]

            yseq = am_scores.argmax(dim=-1)
            score = am_scores.max(dim=-1)[0]
            score = torch.sum(score, dim=-1)

            # pad with mask tokens to ensure compatibility with sos/eos tokens
            yseq = torch.tensor([self.sos] + yseq.tolist() + [self.eos], device=yseq.device)

            nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

            for nbest_idx, hyp in enumerate(nbest_hyps):
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                # Change integer-ids to tokens
                token = tokenizer.ids2tokens(token_int)

                _, timestamp = ts_prediction_lfr6_standard(
                    us_alphas[i][: encoder_out_lens[i] * 3],
                    us_peaks[i][: encoder_out_lens[i] * 3],
                    copy.copy(token),
                    vad_offset=kwargs.get("begin_time", 0),
                )

                text_postprocessed, time_stamp_postprocessed, word_lists = (
                    postprocess_utils.sentence_postprocess(token, timestamp)
                )

                result_i = {
                    "key": key[i],
                    "text": text_postprocessed,
                    "timestamp": time_stamp_postprocessed,
                }

                results.append(result_i)

        time7 = time.perf_counter()
        meta_data["post_process"] = time7 - time6

        return results, meta_data
