# Copyright 2024 Huawei Technologies Co., Ltd
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

"""
Convert alpaca dataset into sharegpt format.

Usage: python convert_alpaca.py --in alpaca_data.json --out alpaca_data_qwen.json
"""

import argparse
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    new_content = []
    for i, c in enumerate(content):
        if len(c["input"].strip()) > 1:
            q, a = c["instruction"] + "\nInput:\n" + c["input"], c["output"]
        else:
            q, a = c["instruction"], c["output"]
        new_content.append(
            {
                "id": f"identity_{i}",
                "conversations": [
                    {"from": "user", "value": q},
                    {"from": "assistant", "value": a},
                ],
            }
        )

    print(f"#out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2, ensure_ascii=False)
