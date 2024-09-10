#!/bin/bash

# 获取当前目录的名字
current_dir=$(basename "$PWD")

# 判断当前目录名字是否为 "finetune"
if [ "$current_dir" = "finetune" ]; then
  # 如果在finetune目录下，则返回上一级目录
  cd ..
fi

source /path/to/cann/ascend-toolkit/set_env.sh

USE_FLASH_ATTENTION_2=true
export use_flash_attention_2=$USE_FLASH_ATTENTION_2

HOSTFILE="finetune/hostfile_16p"
MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')  # 获取hostfile第一行为masteraddr
MASTER_PORT=6001
NODE_ADDR=`hostname -I | awk '{for(i=1;i<=NF;i++)print $i}' | grep ${MASTER_ADDR%.*}. | awk -F " " '{print$1}'`  # 获取本机IP
NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
NNODES=$(cat $HOSTFILE | wc -l)
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
echo $MASTER_ADDR
echo $NODE_ADDR
echo $NODE_RANK
echo $NNODES

MODEL="openbmb/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="path/to/trainging_data"
EVAL_DATA="path/to/test_data"
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune/finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm true \
    --dataloader_num_workers 1 \
    --model_max_length 2048 \
    --max_slice_nums 9 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --output_dir finetune/output/output_minicpmv2 \
    --logging_dir finetune/output/output_minicpmv2 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed finetune/ds_config_zero2.json \
    --report_to "tensorboard" 2>&1 | tee finetune/ds.log 2>&1 &
