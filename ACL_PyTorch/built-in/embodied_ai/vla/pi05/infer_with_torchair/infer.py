# Copyright 2026 Huawei Technologies Co., Ltd
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
import time
import torch
import torch_npu
from lerobot.policies.pi05 import PI05Policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors



def parse_args():
    parser = argparse.ArgumentParser("pi05 infer")
    parser.add_argument("--pi05_model_path", type=str, default="./weight/pi05_libero_finetuned",
                        help="pi05 model checkpoint path")
    parser.add_argument("--tokenizer_path", type=str, default="./weight/paligemma-3b-pt-224",
                        help="paligemma-3b-pt-224 tokenizer path")
    parser.add_argument('--warmup', type=int, default=3, help="Warm up times")
    parser.add_argument('--episode_index', type=int, default=0, help="pick an episode from the dataset")
    args = parser.parse_args()
    return args


def main():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    args = parse_args()
    model_id = args.pi05_model_path
    policy = PI05Policy.from_pretrained(model_id).to(device="npu").eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": 'npu'},
                                "tokenizer_processor": {"tokenizer_name": args.tokenizer_path}}
    )

    dataset = LeRobotDataset("lerobot/libero")

    # pick an episode
    episode_index = args.episode_index

    # each episode corresponds to a contiguous range of frame indices
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]

    # get a single frame from that episode
    frame_index = from_idx
    frame = dict(dataset[frame_index])

    batch = preprocess(frame)

    with torch.inference_mode():
        print(f"----warm up-----")
        for _ in range(args.warmup):
            policy.reset()
            pred_action = policy.select_action(batch)
        print(f"----warm up done-----")
    
        policy.reset()
        t0 = time.time()
        pred_action = policy.select_action(batch)
        pred_action = postprocess(pred_action)
        time_elapsed = time.time() - t0
        print(f"inference duration: {time_elapsed}")
        print(f"action dim: {pred_action.shape}, action vector: {pred_action}")


if __name__ == "__main__":
    main()