#!/usr/bin/env bash

set -e

infer_mode=$1
bs=$2
chip_name=$3
question_seq_len=35 # 取自开源源码 https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/blip_vqa.py#L42

atc \
  --framework 5 \
  --model ascend_models/visual_encoder_md.onnx \
  --output ascend_models/visual_encoder_md \
  --insert_op_conf ../visual_encoder.aippconfig \
  --enable_small_channel 1 \
  --soc_version "Ascend${chip_name}"
atc \
  --framework 5 \
  --model ascend_models/text_encoder_md.onnx \
  --output ascend_models/text_encoder_md \
  --soc_version "Ascend${chip_name}"

if [[ "${infer_mode}" == "rank" ]]; then
  atc \
    --framework 5 \
    --model ascend_models/text_decoder_rank_1_md.onnx \
    --output ascend_models/text_decoder_rank_1_md \
    --soc_version "Ascend${chip_name}"
  atc \
    --framework 5 \
    --model ascend_models/text_decoder_rank_2_md.onnx \
    --output ascend_models/text_decoder_rank_2_md \
    --op_select_implmode 'high_performance' \
    --optypelist_for_implmode 'Gelu' \
    --soc_version "Ascend${chip_name}"
elif [[ "${infer_mode}" == "generate" ]]; then
  atc \
    --framework 5 \
    --model ascend_models/text_decoder_generate.onnx \
    --output ascend_models/text_decoder_generate_sim \
    --input_format 'ND' \
    --input_shape "input_ids:${bs},-1;attention_mask:${bs},-1;encoder_hidden_states:${bs},${question_seq_len},768;encoder_attention_mask:${bs},${question_seq_len}" \
    --dynamic_dims '1,1;2,2;3,3;4,4;5,5;6,6;7,7;8,8;9,9;10,10' \
    --soc_version "Ascend${chip_name}"
    # answer_seq_len 的最大值 10 取自开源源码 https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/blip_vqa.py#L100
fi
