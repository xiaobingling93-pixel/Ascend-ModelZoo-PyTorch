MASTER_ADDR=xxxx
MASTER_PORT=6004
NODE_RANK=0
NNODES=2
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 开启特征值检测，只打印异常日志，不告警
export NPU_ASD_ENABLE=1
source ./test/env_npu.sh

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

torchrun $DISTRIBUTED_ARGS train_multi_with_amp.py --nproc_per_node $GPUS_PER_NODE \
                                          --node_rank $NODE_RANK \
                                          --learning_rate 0.0001 \
                                          --batch_size 1 \
                                          --epochs 1
