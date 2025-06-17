# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

#!/bin/bash

# 默认值
MODEL="mgn_mkt1501_bs1.om"
BATCH_SIZE=1
INPUT_BASE="./Market-1501-v15.09.15"
OUTPUT_BASE="./result"
OUTPUT_FMT="TXT"

# 解析命令行参数
while getopts ":m:b:i:o:f:" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    i) INPUT_BASE="$OPTARG" ;;
    o) OUTPUT_BASE="$OPTARG" ;;
    f) OUTPUT_FMT="$OPTARG" ;;
    \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
    :) echo "选项 -$OPTARG 需要一个参数" >&2; exit 1 ;;
  esac
done

echo "模型文件: $MODEL"
echo "批大小: $BATCH_SIZE"
echo "输入数据路径: $INPUT_BASE"
echo "输出结果路径: $OUTPUT_BASE"
echo "输出格式: $OUTPUT_FMT"

echo "Running inference on q original..."
python3 -m ais_bench --model="$MODEL" --device=0 --batchsize="$BATCH_SIZE" --input="${INPUT_BASE}/bin_data/q/" --output="$OUTPUT_BASE" --output_dirname=q_out --outfmt="$OUTPUT_FMT"

echo "Running inference on g original..."
python3 -m ais_bench --model="$MODEL" --device=0 --batchsize="$BATCH_SIZE" --input="${INPUT_BASE}/bin_data/g/" --output="$OUTPUT_BASE" --output_dirname=g_out --outfmt="$OUTPUT_FMT"

echo "Running inference on q flip..."
python3 -m ais_bench --model="$MODEL" --device=0 --batchsize="$BATCH_SIZE" --input="${INPUT_BASE}/bin_data_flip/q/" --output="$OUTPUT_BASE" --output_dirname=q_filp --outfmt="$OUTPUT_FMT"

echo "Running inference on g flip..."
python3 -m ais_bench --model="$MODEL" --device=0 --batchsize="$BATCH_SIZE" --input="${INPUT_BASE}/bin_data_flip/g/" --output="$OUTPUT_BASE" --output_dirname=g_filp --outfmt="$OUTPUT_FMT"

echo "All inference tasks completed."