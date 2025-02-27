#!/bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
export WORLD_SIZE=8

# start
for ((rank=0; rank<$WORLD_SIZE; rank++))
do
    export LOCAL_RANK=$rank
    python train_8p_shell.py
done

wait