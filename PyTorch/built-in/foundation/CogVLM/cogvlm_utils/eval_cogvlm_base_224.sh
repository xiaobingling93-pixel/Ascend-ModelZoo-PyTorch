#! /bin/bash
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ./env_npu.sh

NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="微调后权重路径"
VERSION="base"
MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length 490 \
    --local_tokenizer 分词器权重路径 \
    --version $VERSION"
# Tips: If training models of resolution 244, you can set --max_length smaller 


OPTIONS_SAT="SAT_HOME=~/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

test_data="./archive_split/test"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 0 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --test-data ${test_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --strict-eval \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config test_config_bf16.json \
       --skip-init \
       --seed 1234
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 18888 --hostfile ${HOST_FILE_PATH} evaluate_cogvlm_demo.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x