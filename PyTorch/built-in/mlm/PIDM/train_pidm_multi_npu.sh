source /path_to_cann/set_env.sh

# 不使用wandb
export WANDB_DISABLED=true
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -m torch.distribute.run --nproc_per_node=8 --master_port 48940 train.py \
--dataset_path "./dataset/deepfashion" --batch_size 14 --use_bf16 --exp_name "pidm_deepfashion"