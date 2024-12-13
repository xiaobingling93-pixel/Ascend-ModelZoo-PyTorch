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

import torch
import mindietorch

from mindie_paraformer import precision_eval
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.models.ct_transformer.model import CTTransformer
from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words


class MindiePunc(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
    
    def forward(self, text, text_lengths):
        y, _ = self.model.punc_forward(text, text_lengths)
        _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
        punctuations = torch.squeeze(indices, dim=1)

        return punctuations

    @staticmethod
    def export(punc, path="./compiled_punc.pt", soc_version="Ascendxxx"):
        print("Begin tracing punc model.")

        input_shape = (1, 20)
        min_shape = (1, -1)
        max_shape = (1, -1)
        input_speech = torch.randint(1, 10, input_shape, dtype=torch.int32)
        input_speech_lengths = torch.tensor([20, ], dtype=torch.int32)
        compile_inputs = [mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=torch.int32),
                          mindietorch.Input(min_shape=(1, ), max_shape=(1, ), dtype=torch.int32)]
        
        export_model = torch.jit.trace(punc, example_inputs=(input_speech, input_speech_lengths))
        print("Finish tracing punc model.")

        compiled_model = mindietorch.compile(
            export_model,
            inputs=compile_inputs,
            precision_policy=mindietorch.PrecisionPolicy.PREF_FP16,
            default_buffer_size_vec=[10, ],
            soc_version=soc_version,
            ir="ts"
        )
        compiled_model.save(path)
        print("Finish compiling punc model, compiled model is saved in {}.".format(path))

        print("Start checking the percision of punc model.")
        sample_speech = torch.randint(1, 10, (1, 10), dtype=torch.int32)
        sample_speech_lengths = torch.tensor([10, ], dtype=torch.int32)
        mrt_res = compiled_model(sample_speech.to("npu"), sample_speech_lengths.to("npu"))
        ref_res = punc(sample_speech, sample_speech_lengths)
        precision_eval(mrt_res, ref_res)


class MindieCTTransformer(CTTransformer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mindie_punc = torch.jit.load(kwargs["compiled_punc"])
    
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        assert len(data_in) == 1
        text = load_audio_text_image_video(data_in, data_type=kwargs.get("kwargs", "text"))[0]

        split_size = kwargs.get("split_size", 20)

        tokens = split_words(text, jieba_usr_dict=self.jieba_usr_dict)
        tokens_int = tokenizer.encode(tokens)

        mini_sentences = split_to_mini_sentence(tokens, split_size)
        mini_sentences_id = split_to_mini_sentence(tokens_int, split_size)
        assert len(mini_sentences) == len(mini_sentences_id)

        mini_sentences_id = [torch.unsqueeze(torch.tensor(id, dtype=torch.int32), 0) for id in mini_sentences_id]

        cache_sent = []
        cache_sent_id = torch.tensor([[]], dtype=torch.int32)
        new_mini_sentence = ""
        cache_pop_trigger_limit = 200
        results = []
        meta_data = {}

        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = torch.cat([cache_sent_id, mini_sentence_id], dim=1)

            text = mini_sentence_id.to("npu")
            text_lengths = torch.tensor([text.shape[1], ], dtype=torch.int32).to("npu")
            punctuations = self.mindie_punc(text, text_lengths)
            punctuations = punctuations.to("cpu")

            assert punctuations.size()[0] == len(mini_sentence)

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if (
                        self.punc_list[punctuations[i]] == "。"
                        or self.punc_list[punctuations[i]] == "？"
                    ):
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if (
                    sentenceEnd < 0
                    and len(mini_sentence) > cache_pop_trigger_limit
                    and last_comma_index >= 0
                ):
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.sentence_end_id
                cache_sent = mini_sentence[sentenceEnd + 1 :]
                cache_sent_id = mini_sentence_id[:, sentenceEnd + 1 :]
                mini_sentence = mini_sentence[0 : sentenceEnd + 1]
                punctuations = punctuations[0 : sentenceEnd + 1]

            words_with_punc = []
            for i in range(len(mini_sentence)):
                if (
                    i == 0
                    or self.punc_list[punctuations[i - 1]] == "。"
                    or self.punc_list[punctuations[i - 1]] == "？"
                ) and len(mini_sentence[i][0].encode()) == 1:
                    mini_sentence[i] = mini_sentence[i].capitalize()
                if i == 0:
                    if len(mini_sentence[i][0].encode()) == 1:
                        mini_sentence[i] = " " + mini_sentence[i]
                if i > 0:
                    if (
                        len(mini_sentence[i][0].encode()) == 1
                        and len(mini_sentence[i - 1][0].encode()) == 1
                    ):
                        mini_sentence[i] = " " + mini_sentence[i]
                words_with_punc.append(mini_sentence[i])
                if self.punc_list[punctuations[i]] != "_":
                    punc_res = self.punc_list[punctuations[i]]
                    if len(mini_sentence[i][0].encode()) == 1:
                        if punc_res == "，":
                            punc_res = ","
                        elif punc_res == "。":
                            punc_res = "."
                        elif punc_res == "？":
                            punc_res = "?"
                    words_with_punc.append(punc_res)
            new_mini_sentence += "".join(words_with_punc)
            # Add Period for the end of the sentence
            new_mini_sentence_out = new_mini_sentence
            if mini_sentence_i == len(mini_sentences) - 1:
                if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                elif new_mini_sentence[-1] == ",":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "."
                elif (
                    new_mini_sentence[-1] != "。"
                    and new_mini_sentence[-1] != "？"
                    and len(new_mini_sentence[-1].encode()) != 1
                ):
                    new_mini_sentence_out = new_mini_sentence + "。"
                    if len(punctuations):
                        punctuations[-1] = 2
                elif (
                    new_mini_sentence[-1] != "."
                    and new_mini_sentence[-1] != "?"
                    and len(new_mini_sentence[-1].encode()) == 1
                ):
                    new_mini_sentence_out = new_mini_sentence + "."
                    if len(punctuations):
                        punctuations[-1] = 2

        result_i = {"key": key[0], "text": new_mini_sentence_out}
        results.append(result_i)
        return results, meta_data
