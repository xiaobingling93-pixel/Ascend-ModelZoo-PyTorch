#!/bin/bash -l
##################基础参数，需要模型审视修改###################
DIR=`pwd`

# 采集日志
## 将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
## 设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
## 设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
## 设置是否开启taskque,0-关闭/1-开启
export TASK_QUEUE_ENABLE=1
## HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1

# 网络名称，同目录名称
Network="qwen_7b"

# 预训练模型
model_name="./model_weight"

# deepspeed配置
deepspeed_config="./zero1.json"
output_path="./output"

# 网络参数配置
batch_size=1
gradient_accumulation_steps=1
epochs=100.0
cutoff=8192

# 分布式参数
GPUS_PER_NODE=8 #$(python -c 'import torch; import torch_npu; print(torch.npu.device_count())')
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# 参数传入
for para in $*
do
    if [[ $para == --model_name* ]];then
        model_name=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --gradient_accumulation_steps* ]];then
        gradient_accumulation_steps=`echo ${para#*=}`
    elif [[ $para == --cutoff* ]];then
        cutoff=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    elif [[ $para == --nodes* ]];then
        NNODES=`echo ${para#*=}`
    elif [[ $para == --NODE_RANK* ]];then
        NODE_RANK=`echo ${para#*=}`
    elif [[ $para == --master_addr* ]];then
        MASTER_ADDR=`echo ${para#*=}`
    elif [[ $para == --master_port* ]];then
        MASTER_PORT=`echo ${para#*=}`
    elif [[ $para == --deepspeed_config* ]];then
        deepspeed_config=`echo ${para#*=}`
    fi
done

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

if [ -d ${output_path} ];then
    echo "output path ${output_path} has already existed."
else
    mkdir -p ${output_path}
fi
timestamp=$(date +"%Y%m%d-%H%M%S")

echo "start to run ${Network}, check log from ${output_path}"
nohup torchrun $DISTRIBUTED_ARGS src/train_bash.py \
    --deepspeed ${deepspeed_config} \
    --stage pt \
    --model_name_or_path ${model_name} \
    --use_fast_tokenizer False \
    --do_train \
    --dataset wiki_demo  \
    --cutoff ${cutoff} \
    --finetuning_type full \
    --output_dir ${output_path} \
    --ddp_timeout 36000000 \
    --overwrite_output_dir \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10000 \
    --learning_rate 1e-5 \
    --num_train_epochs ${epochs} \
    --plot_loss \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --preprocessing_num_workers 20 \
    --save_total_limit 1 | tee ${output_path}/train_${NODE_RANK}_${timestamp}.log &
    