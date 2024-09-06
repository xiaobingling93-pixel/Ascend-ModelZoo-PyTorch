# Copyright 2023 Huawei Technologies Co., Ltd
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
from config import InferrnceConfig
from inference import LlamaInterface

def main(cli: bool, engine: LlamaInterface, dataset):
    if cli:
        if dataset == 'BoolQ':
            engine.test_boolq()
        elif dataset == 'CEval':
            engine.test_ceval()
        elif dataset == 'GSM8K':
            engine.test_gsm8k()
        else:
            print("dataset is not support! ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cli', dest='cli', default=False, action='store_true',
        help="run web ui by default, if add --cli, run cli."
    )
    parser.add_argument("--kv_size", type=int, default=1024)
    parser.add_argument(
        "--engine", type=str, default="acl",
        help="inference backend, onnx or acl"
    )
    parser.add_argument(
        "--sampling", type=str, default="top_k",
        help="sampling method, greedy, top_k or top_p"
    )
    parser.add_argument(
        "--sampling_value", type=float,default=10,
        help="if sampling method is seted to greedy, this argument will be ignored; if top_k, it means value of p; if top_p, it means value of p"
    )
    parser.add_argument(
        "--temperature", type=float,default=0.7,
        help="sampling temperature if sampling method is seted to greedy, this argument will be ignored."
    )
    parser.add_argument(
        "--hf-dir", type=str, default="/root/model/tiny-llama-1.1B", 
        help="path to huggingface model dir"
    )
    parser.add_argument(
        "--model", type=str, default="/root/model/tiny-llama-seq-1-key-256-int8.om", 
        help="path to onnx or om model"
    )
    parser.add_argument(
        "--dataset", type=str, default="BoolQ"
    )
    
    args = parser.parse_args()
    cfg = InferenceConfig(
        hf_model_dir=args.hf_dir,
        model=args.model,
        max_cache_size=args.kv_size,
        sampling_method=args.sampling,
        sampling_value=args.sampling_value,
        temperature=args.temperature,
        session_type=args.engine,
    )
    engine = LlamaInterface(cfg)
    main(args.cli,engine,args.dataset)