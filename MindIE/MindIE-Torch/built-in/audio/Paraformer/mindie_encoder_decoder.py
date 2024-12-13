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
import os
sys.path.append("./FunASR")

import torch
import mindietorch


class MindieEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
    
    def forward(self, speech, speech_length):
        batch = {"speech": speech, "speech_lengths": speech_length}
        enc, enc_len = self.model.encoder(**batch)
        mask = self.model.make_pad_mask(enc_len)[:, None, :]
        hidden, alphas, pre_token_length = self.model.predictor(enc, mask)
        return enc, hidden, alphas, pre_token_length

    @staticmethod
    def export_ts(encoder, path="./compiled_encoder.pt", soc_version="Ascendxxx", traced_path=None):
        print("Begin tracing encoder.")

        input_shape = (2, 50, 560)
        min_shape = (-1, -1, 560)
        max_shape = (-1, -1, 560)

        if traced_path is not None and os.path.exists(traced_path):
            export_model = torch.load(traced_path)
            print("Load existing traced encoder from {}".format(traced_path))
        else:
            input_speech = torch.randn(input_shape, dtype=torch.float32)
            input_speech_lens = torch.tensor([50, 25], dtype=torch.int32)
        
            export_model = torch.jit.trace(encoder, example_inputs=(input_speech, input_speech_lens))
            print("Finish tracing encoder.")

        compile_inputs = [mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=torch.float32),
                    mindietorch.Input(min_shape=(-1, ), max_shape=(-1, ), dtype=torch.int32)]

        compiled_model = mindietorch.compile(
            export_model,
            inputs=compile_inputs,
            precision_policy=mindietorch.PrecisionPolicy.PREF_FP16,
            default_buffer_size_vec=[400, 1, 400, 1],
            soc_version=soc_version,
            ir="ts"
        )
        compiled_model.save(path)
        print("Finish compiling encoder, compiled model is saved in {}.".format(path))

        print("Start checking the percision of encoder.")
        sample_speech = torch.randn((4, 100, 560), dtype=torch.float32)
        sample_speech_lens = torch.tensor([100, 50, 100, 25], dtype=torch.int32)
        _ = compiled_model(sample_speech.to("npu"), sample_speech_lens.to("npu"))
        print("Finish checking encoder.")


class MindieDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
    
    def forward(self, encoder_out, encoder_out_lens, sematic_embeds, pre_token_length):
        decoder_outs = self.model.decoder(encoder_out, encoder_out_lens, sematic_embeds, pre_token_length)
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)

        encoder_out_mask = self.model.make_pad_mask(encoder_out_lens)[:, None, :]

        us_alphas = self.model.predictor.get_upsample_timestamp(encoder_out, encoder_out_mask, pre_token_length)

        return decoder_out, us_alphas

    @staticmethod
    def export_ts(decoder, path="./compiled_decoder.pt", soc_version="Ascendxxx", traced_path=None):
        print("Begin tracing decoder.")

        input_shape1 = (2, 939, 512)
        min_shape1 = (-1, -1, 512)
        max_shape1 = (-1, -1, 512)

        input_shape2 = (2, 261, 512)
        min_shape2 = (-1, -1, 512)
        max_shape2 = (-1, -1, 512)

        if traced_path is not None and os.path.exists(traced_path):
            export_model = torch.load(traced_path)
            print("Load existing traced decoder from {}".format(traced_path))
        else:
            encoder_out = torch.randn(input_shape1, dtype=torch.float32)
            encoder_out_lens = torch.tensor([939, 500], dtype=torch.int32)
            sematic_embeds = torch.randn(input_shape2, dtype=torch.float32)
            sematic_embeds_lens = torch.tensor([261, 100], dtype=torch.int32)
            
            export_model = torch.jit.trace(decoder, example_inputs=(encoder_out, encoder_out_lens, sematic_embeds, sematic_embeds_lens))
            print("Finish tracing decoder.")

        compile_inputs = [mindietorch.Input(min_shape=min_shape1, max_shape=max_shape1, dtype=torch.float32),
            mindietorch.Input(min_shape=(-1, ), max_shape=(-1, ), dtype=torch.int32),
            mindietorch.Input(min_shape=min_shape2, max_shape=max_shape2, dtype=torch.float32),
            mindietorch.Input(min_shape=(-1, ), max_shape=(-1, ), dtype=torch.int32)]

        compiled_model = mindietorch.compile(
            export_model,
            inputs=compile_inputs,
            precision_policy=mindietorch.PrecisionPolicy.PREF_FP16,
            default_buffer_size_vec=[800, 10],
            soc_version=soc_version,
            ir="ts"
        )
        compiled_model.save(path)
        print("Finish compiling decoder, compiled model is saved in {}.".format(path))

        print("Start checking the percision of decoder.")
        sample_encoder = torch.randn((4, 150, 512), dtype=torch.float32)
        sample_encoder_lens = torch.tensor([150, 100, 150, 50], dtype=torch.int32)
        sample_sematic = torch.randn((4, 50, 512), dtype=torch.float32)
        sample_sematic_lens = torch.tensor([50, 30, 50, 10], dtype=torch.int32)
        _ = compiled_model(sample_encoder.to("npu"), sample_encoder_lens.to("npu"), sample_sematic.to("npu"), sample_sematic_lens.to("npu"))
        print("Finish checking decoder.")