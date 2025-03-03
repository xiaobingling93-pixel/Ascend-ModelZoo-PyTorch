#!/bin/bash

export MASTER_ADDR="xxxx"  # change to your master node IP address
export MASTER_PORT="12345"
export WORLD_SIZE=16
export NODE_RANK=1

# start
for ((local_rank=0; local_rank<8; local_rank++))
do
    export RANK=$((NODE_RANK * 8 + local_rank))
    export LOCAL_RANK=$local_rank
    python train_16p_shell.py &

done

wait