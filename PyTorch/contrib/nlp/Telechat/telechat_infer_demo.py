# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch_npu
from torch_npu.contrib import transfer_to_npu

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = sys.argv[1]

torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)

def main(PATH):
    # 加载模型相关
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True).half().npu()
    generate_config = GenerationConfig.from_pretrained(PATH)
    model.eval()

    #  chat(bot)模型多轮演示
    print("*" * 10 + "多轮输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    answer, history = model.chat(tokenizer = tokenizer, question=question, history=[], generation_config=generate_config,
                                 stream=False)
    print("回答:", answer)
    print("截至目前的聊天记录是:", history)

    question = "你是谁训练的"
    print("提问:", question)
    # 将history传入
    answer, history = model.chat(tokenizer, question=question, history=history, generation_config=generate_config,
                                 stream=False)
    print("回答是:", answer)
    print("截至目前的聊天记录是:", history)


if __name__ == '__main__':
    main(PATH)

