#!/bin/bash
source ./test/env_npu.sh

# model inference config
tokenizer_path="YOUR_PATH"
model_name_or_path="YOUR_PATH"
generation_config_path="YOUR_PATH"
output_path="./output"
device_map="npu"
seed=1234


# infer
python infer.py --tokenizer_path=$tokenizer_path \
--model_name_or_path=$model_name_or_path \
--generation_config_path=$generation_config_path \
--output_path=$output_path \
--device_map=$device_map \
--bf16 \
--seed=$seed