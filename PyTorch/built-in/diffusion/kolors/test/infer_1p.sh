#!/bin/bash
source ./test/env_npu.sh

# model inference config
ckpt_dir="YOUR_PATH"
prompt="YOUR_PROMPT"
device_map="npu"
output_path="./output"
seed=66


# infer
python3 infer.py --ckpt_dir=$tokenizer_path \
--prompt=$prompt \
--device_map=$device_map \
--output_path=$output_path \
--seed=$seed
