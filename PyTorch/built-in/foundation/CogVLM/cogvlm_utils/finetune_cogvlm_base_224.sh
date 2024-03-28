#! /bin/bash
source ./env_npu.sh
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="预训练权重路径"
VERSION="base"
MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length 512 \
    --local_tokenizer 分词器权重路径 \
    --version $VERSION"
# Tips: If training models of resolution 244, you can set --max_length smaller 

OPTIONS_SAT="SAT_HOME=~/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

# 用户需要指定train_data和valid_data为实际路径，例如"../archive_split/train"和"../archive_split/valid"
train_data="训练数据路径"
valid_data="验证数据路径"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 1000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${valid_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --log-interval 1 \
       --save-interval 2000 \
       --eval-interval 200 \
       --save "./checkpoints" \
       --eval-iters 10 \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config test_config_bf16.json \
       --skip-init \
       --seed 1234 
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_cogvlm_demo.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
