import os
import torch
import sys
import platform
import signal
import argparse
import torch_aie
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.expanduser("/root/.cache/huggingface/modules"))
model_pth = "./model/"

tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)
model = AutoModel.from_pretrained(model_pth, trust_remote_code=True, torchscript=True).float()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu", help="cpu/npu")
    args = parser.parse_args()
    return args

def main():
    past_key_values, history = None, []
    global stop_stream
    args = parse_arg()
    device = args.device
    print("device:", device)
    batch_size = args.batch_size
    aie_model = None
    if device == "npu":
        torch_aie.set_device(0)
        aie_model_path = "./chatglm2_6b_batch_1_compiled.ts"
        aie_model = torch.jit.load(aie_model_path)
        aie_model.eval()
    
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        result = model.stream_chat(tokenizer, aie_model, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True)
        for response, history, past_key_values in result:
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()