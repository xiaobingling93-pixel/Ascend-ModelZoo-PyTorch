#!/bin/bash

export WANDB_DISABLED=true

project_dir=$(cd "$(dirname $0)"; pwd)

model="./model"
data_path="finetune/data.json"
exp_id="0"

# 参数赋值
for para in $*
do
    if [[ $para == --model* ]];then
        model=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --exp_id* ]];then
        exp_id=`echo ${para#*=}`
    fi
done

output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed_args="--master_port=$((10000 + RANDOM % 20000))"

taskset -c 0-63 deepspeed ${deepspeed_args} ${project_dir}/finetune.py \
    --deepspeed ${project_dir}/ds_config_zero2.json \
    --model_name_or_path ${model} \
    --data_path ${data_path} \
    --model_max_length 512 \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --learning_rate 2e-6 \
    --max_steps 2000 \
    --save_steps -1 \
    --dataloader_num_workers 64 \
    --dataloader_persistent_workers True \
    --bf16 > ${log_dir}/train.log 2>&1 &
wait

#结果打印，不需要修改
echo "------------------ Final result ------------------"
train_samples_per_second=$(grep -oP 'train_samples_per_second.*?\K\d+\.\d+' ${log_dir}/train.log)
echo "Train samples per second : $train_samples_per_second"
