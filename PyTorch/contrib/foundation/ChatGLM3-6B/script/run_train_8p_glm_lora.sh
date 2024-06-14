#!/usr/bin/env bash
clear
export IS_USE_FLASH=1
export INF_NAN_MODE_ENABLE=0
export OMP_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=7200
export ACLNN_CACHE_LIMIT=100000
torchrun --standalone --nnodes=1 --nproc_per_node=8 ./script/finetune_hf.py ./data/AdvertiseGen_fix/ ./models/ ./configs/lora.yaml