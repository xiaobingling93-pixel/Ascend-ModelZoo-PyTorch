#!/bin/bash

source groundingdino_npu/env_npu.sh
PYTHON_PATH="Python Env Path"
export Mx_Driving_PYTHON_PATH=${PYTHON_PATH}/lib/python3.8
export ASCEND_CUSTOM_OPP_PATH=${Mx_Driving_PYTHON_PATH}/site-packages/mx_driving/packages/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

CONFIG=groundingdino_npu/mm_grounding_dino_swin-b_finetune_b2_refcoco.py
GPUS=8
NNODES=1
NODE_RANK=0
PORT=6005
MASTER_ADDR=localhost

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_npu.py \
    $CONFIG \
    --launcher pytorch ${@:3}
