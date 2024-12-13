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

import argparse
import sys
sys.path.append("./FunASR")

import torch

from funasr.auto.auto_model import AutoModel


torch.library.define("aie::flash_attention", "(Tensor query, Tensor key, Tensor value, int num_head, "
        "Tensor? attn_mask=None, Tensor? pse=None, float scale=1.0, str layout='BSH', str type='PFA') -> Tensor")


@torch.library.impl('aie::flash_attention', "cpu")
def flash_attention_wrapper(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, num_head: int,
                            attn_mask: torch.Tensor = None, pse: torch.Tensor = None, scale: float = 1.0,
                            layout: str = 'BSH', type: str = 'PFA') -> torch.Tensor:
    return query


class ParaformerEncoder(torch.nn.Module):
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
    def trace_model(encoder, path="./traced_encoder.pt"):
        print("Begin trace encoder!")

        input_shape = (2, 50, 560)
        input_speech = torch.randn(input_shape, dtype=torch.float32)
        input_speech_lens = torch.tensor([50, 25], dtype=torch.int32)
        
        trace_model = torch.jit.trace(encoder, example_inputs=(input_speech, input_speech_lens))
        trace_model.save(path)
        print("Finish trace encoder")


class ParaformerDecoder(torch.nn.Module):
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
    def trace_model(decoder, path="./traced_decoder.pt"):
        print("Begin trace decoder!")

        input_shape1 = (2, 939, 512)
        input_shape2 = (2, 261, 512)

        encoder_out = torch.randn(input_shape1, dtype=torch.float32)
        encoder_out_lens = torch.tensor([939, 500], dtype=torch.int32)
        sematic_embeds = torch.randn(input_shape2, dtype=torch.float32)
        sematic_embeds_lens = torch.tensor([261, 100], dtype=torch.int32)
    
        trace_model = torch.jit.trace(decoder, example_inputs=(encoder_out, encoder_out_lens, sematic_embeds, sematic_embeds_lens))
        trace_model.save(path)
        print("Finish trace decoder")


class AutoModelParaformer(AutoModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def trace(**kwargs):
        model, kwargs = AutoModel.build_model(**kwargs)
        
        import copy
        from funasr.models.bicif_paraformer.export_meta import export_rebuild_model

        kwargs_new = copy.deepcopy(kwargs)
        kwargs_new['onnx'] = False
        kwargs_new["max_seq_len"] = 512
        del kwargs_new["model"]
        model = export_rebuild_model(model, **kwargs_new)

        encoder = ParaformerEncoder(model)
        ParaformerEncoder.trace_model(encoder, kwargs["traced_encoder"])

        decoder = ParaformerDecoder(model)
        ParaformerDecoder.trace_model(decoder, kwargs["traced_decoder"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model",
                        help="path of pretrained model")
    parser.add_argument("--traced_encoder", default="./compiled_model/traced_encoder.pt",
                        help="path to save compiled decoder")
    parser.add_argument("--traced_decoder", default="./compiled_model/traced_decoder.pt",
                        help="path to save compiled decoder")
    args = parser.parse_args()

    AutoModelParaformer.trace(model=args.model, traced_encoder=args.traced_encoder, traced_decoder=args.traced_decoder)