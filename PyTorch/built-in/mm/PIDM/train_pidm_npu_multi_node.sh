source /path_to_cann/set_env.sh

export COMBINED_ENABLE=1
export OMP_NUM_THREADS=1
export CPU_AFFINITY_CONF=1

# 不使用wandb
export WANDB_DISABLED=true
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export MASTER_ADDR=""
export MASTER_PORT=48940
export NODE_RANK=0

python -m torch.distributed.run --nproc_per_node=8 --nnode=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py \
--dataset_path "./dataset/deepfashion" --batch_size 14 --use_bf16 --exp_name "pidm_deepfashion"