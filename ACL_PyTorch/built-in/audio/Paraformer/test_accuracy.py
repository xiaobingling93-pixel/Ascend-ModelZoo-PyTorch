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


import argparse
from tqdm import tqdm
from nltk.metrics.distance import edit_distance


def load_text(file_name):
    results = {}
    with open(file_name, "r") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                results[parts[0]] = parts[1]
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./aishell_test_result.txt", type=str,
                        help="path to saved infer result")
    parser.add_argument("--ref_path", default="/path/to/AISHELL-1/transcript/aishell_transcript_v0.8.txt",
                        type=str, help="directory or path of sample audio")
    args = parser.parse_args()

    infer_result = load_text(args.result_path)
    ref_result = load_text(args.ref_path)

    infer_list = []
    refer_list = []
    for key, value in infer_result.items():
        if key in ref_result:
            infer_list.append(value.replace(" ", ""))
            refer_list.append(ref_result[key].replace(" ", ""))

    cer_total = 0
    step = 0
    for infer, refer in tqdm(zip(infer_list, refer_list)):
        infer = [i for i in infer]
        refer = [r for r in refer]
        cer_total += edit_distance(infer, refer) / len(refer)
        step += 1
    
    cer = cer_total / step
    accuracy = 1 - cer
    print("character-errer-rate: {:.4f}, accuracy: {:.4f}".format(cer, accuracy))