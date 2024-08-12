#!/usr/bin/env bash

set -e

infer_mode=$1
bs=$2
question_seq_len=35 # 取自开源源码 https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/blip_vqa.py#L42
k_test=128 # 取自开源配置文件 https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/configs/vqa.yaml#L19

process_model() {
  local model_name=$1
  shift
  local params=("$@")

  local input_path="ascend_models/${model_name}.onnx"
  local output_path="ascend_models/${model_name}_sim.onnx"

  for ((i=0; i<${#params[@]}; i+=2)); do
    local param_name=${params[i]}
    local param_value=${params[i+1]}

    python3 -m onnxruntime.tools.make_dynamic_shape_fixed \
    --dim_param "${param_name}" \
    --dim_value "${param_value}" \
    "${input_path}" \
    "${output_path}"

    input_path="${output_path}"
  done

  onnxsim "${input_path}" "${output_path}"
}

process_model 'visual_encoder' 'bs' "${bs}" 'question_seq_len' "${question_seq_len}"
process_model 'text_encoder' 'bs' "${bs}" 'question_seq_len' "${question_seq_len}"
if [[ "${infer_mode}" == "rank" ]]; then
  process_model 'text_decoder_rank_1' 'bs' "${bs}" 'question_seq_len' "${question_seq_len}"
  process_model 'text_decoder_rank_2' 'bs' "${bs}" 'question_seq_len' "${question_seq_len}" 'bs*k_test' "$((bs*k_test))"
elif [[ "${infer_mode}" == "generate" ]]; then
  process_model 'text_decoder_generate' 'bs' "${bs}" 'question_seq_len' "${question_seq_len}"
fi
