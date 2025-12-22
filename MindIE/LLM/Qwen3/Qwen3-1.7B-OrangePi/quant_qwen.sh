#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --model_path    必填，模型源路径"
    echo "  --save_directory 必填，量化后保存路径"
    echo "  --start-layer   可选，回退层起始编号（默认0）"
    echo "  --end-layer     可选，回退层结束编号（默认27）"
    echo "  --help          显示帮助"
    echo ""
    echo "示例: $0 --model_path /path/to/Qwen3-1.7B --save_directory /path/to/save"
}

START_LAYER=0
END_LAYER=27
MODEL_PATH=""
SAVE_DIRECTORY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --save_directory)
            SAVE_DIRECTORY="$2"
            shift 2
            ;;
        --start-layer)
            START_LAYER="$2"
            shift 2
            ;;
        --end-layer)
            END_LAYER="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "错误：未知参数 $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" || -z "$SAVE_DIRECTORY" ]]; then
    echo "错误：--model_path 和 --save_directory 为必填参数！"
    usage
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "错误：模型路径 $MODEL_PATH 不存在！"
    exit 1
fi

DISABLE_NAMES=("lm_head")
for ((i=START_LAYER; i<=END_LAYER; i++)); do
    DISABLE_NAMES+=("model.layers.${i}.mlp.down_proj")
done
DISABLE_NAMES_STR="${DISABLE_NAMES[*]}"

python3 msit/msmodelslim/example/Qwen/quant_qwen.py \
  --model_path "$MODEL_PATH" \
  --save_directory "$SAVE_DIRECTORY" \
  --calib_file msit/msmodelslim/example/common/boolq.jsonl \
  --w_bit 8 \
  --a_bit 8 \
  --device_type npu \
  --model_type qwen3 \
  --disable_names ${DISABLE_NAMES_STR} \
  --anti_method m6
