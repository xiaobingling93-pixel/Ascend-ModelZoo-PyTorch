#!/bin/bash

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

MODEL="/your/path/Qwen-7B-Chat" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="../alpaca_data_qwen.json"

cur_path=`pwd`
test_path_dir=${cur_path}
source ${test_path_dir}/env_npu.sh

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/ ];then
    rm -rf ${test_path_dir}/output/
    mkdir -p ${test_path_dir}/output/
else
    mkdir -p ${test_path_dir}/output/
fi

if [ -d ${test_path_dir}/output_qwen/ ];then
    rm -rf ${test_path_dir}/output_qwen/
    mkdir -p ${test_path_dir}/output_qwen/
else
    mkdir -p ${test_path_dir}/output_qwen/
fi

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 12345
"

#训练开始时间，不需要修改
start_time=$(date +%s)

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir output_qwen \
    --max_steps 2000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ./ds_config_zero2.json > ./output/finetune_7B_Chat.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

##################获取训练数据################
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能Train_Samples_Per_Second，需要模型审视修改
Train_Samples_Per_Second=`grep -a 'train_samples_per_second' ${test_path_dir}/output/finetune_1_8B_Chat.log|awk -F "'train_samples_per_second':" '{print $2}'|awk -F "," '{print $1}'`
#打印，不需要修改
echo "Train Samples Per Second : $Train_Samples_Per_Second"
echo "E2E Training Duration sec : $e2e_time"
