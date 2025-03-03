#!/bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
export WORLD_SIZE=8

# start
for ((local_rank=0; local_rank<$WORLD_SIZE; local_rank++))
do
    export LOCAL_RANK=$local_rank
    python train_8p_shell.py
    
done

wait