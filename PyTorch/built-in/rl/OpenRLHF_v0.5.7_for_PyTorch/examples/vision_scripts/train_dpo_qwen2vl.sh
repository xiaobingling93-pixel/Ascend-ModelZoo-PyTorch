set -x

export ACLNN_CACHE_LIMIT=100000
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export HF_DATASETS_OFFLINE=1

read -r -d '' training_commands <<EOF
openrlhf.cli.train_vl_dpo \
   --task_type dpo \
   --save_path ./checkpoint/qwen2vl_dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --model_arch qwen2_vl \
   --pretrain ./Qwen2-VL-2B-Instruct \
   --bf16 \
   --max_epochs 3 \
   --max_len 4096 \
   --zero_stage 2 \
   --learning_rate 1e-7 \
   --lr_scheduler constant \
   --beta 0.1 \
   --dataset dataset/RLHF-V \
   --dataset_config_path examples/vision_scripts/rlhf_v.json \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn sdpa \
   --load_checkpoint \
   --processing_num_workers 16 \
   --gradient_checkpointing
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
