#! /bin/bash
source ./env_npu.sh

NUM_GPUS_PER_WORKER=8
MASTER_PORT=16668
DATASET_PATH="训练数据路径"
MODEL_PATH="预训练权重路径"
SAVE_PATH="模型保存路径"


gpt_options=" \
       --ds_config ds_config.yaml \
       --model_path ${MODEL_PATH} \
       --dataset_path ${DATASET_PATH} \
       --save_path ${SAVE_PATH} \
       --num_epochs 2 \
       --batch_size 1 \
       --save_step 5000000 \
       --lora_rank ${NUM_GPUS_PER_WORKER} \
       --lora_dropout 0.0 \
       --lr 1e-6 \
       --torch_type torch.bfloat16
"

deepspeed --master_port ${MASTER_PORT} --num_gpus ${NUM_GPUS_PER_WORKER} peft_lora.py ${gpt_options} | tee train_cogvlm2.log 2>&1
