#!/bin/bash

cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"scripts" ];then
    scripts_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    scripts_path_dir=${cur_path}/scripts
fi

#创建输出目录，不需要修改
if [ -d ${scripts_path_dir}/output ];then
    rm -rf ${scripts_path_dir}/output
    mkdir -p ${scripts_path_dir}/output
else
    mkdir -p ${scripts_path_dir}/output
fi

#配置NPU环境
source ${scripts_path_dir}/env_npu.sh

# 启动训练脚本
start_time=$(date +%s)
deepspeed --num_gpus 8 --master_port 9910 run_seq2seq_qa.py \
  --model_name_or_path  google/flan-t5-xxl \
  --deepspeed ds_config.json \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-6 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --max_steps 2000 \
  --max_seq_length 384 \
  --logging_steps 1 \
  --doc_stride 128 \
  --seed 1234 \
  --bf16 \
  --overwrite_output_dir \
  --output_dir ./output > ${scripts_path_dir}/output/run_flan_t5_xxl.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"