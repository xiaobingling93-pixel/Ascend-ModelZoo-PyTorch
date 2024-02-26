#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1200
export INF_NAN_MODE_ENABLE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MASTER_PORT=6669

MODEL_PATH="../model"

HCCL_CONNECT_TIMEOUT=1200  deepspeed  --num_gpus ${NUM_GPUS_PER_WORKER}  src/train_bash.py \
    --stage sft \
    --model_name_or_path  $MODEL_PATH \
    --deepspeed ./ds_config_zero3.json \
    --do_train \
    --dataset alpaca_gpt4_en,alpaca_gpt4_zh \
    --template default \
    --finetuning_type full \
    --output_dir ./output_sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 10000000 \
    --learning_rate 1e-6 \
    --num_train_epochs 5.0 \
    --max_grad_norm 0.5 \
    --plot_loss \
    --fp16 | tee logs/train.log 
