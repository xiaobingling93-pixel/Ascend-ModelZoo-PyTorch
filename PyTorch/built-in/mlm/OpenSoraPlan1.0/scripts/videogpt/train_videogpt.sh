
Network="VideoGPT"

source ./scripts/env_npu.sh

output_path=./scripts/videogpt/output/

mkdir -p ${output_path}

start_time=$(date +%s)
echo "start_time: ${start_time}"

accelerate launch \
  --config_file scripts/accelerate_configs/ddp_config.yaml \
  opensora/train/train_videogpt.py \
  --do_train \
  --seed 1234 \
  --data_path "/MSRVTT/" \
  --per_device_train_batch_size 1  \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --weight_decay 0. \
  --max_steps 2000 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --save_strategy steps \
  --save_total_limit 5 \
  --logging_steps 1 \
  --save_steps 1000 \
  --n_codes 2048 \
  --n_hiddens 256 \
  --embedding_dim 4 \
  --n_res_layers 4 \
  --downsample "4,4,4" \
  --resolution 256 \
  --sequence_length 16 \
  --output_dir results/videogpt_488_256_16 \
  --bf16 True \
  --fp16 False \
  --report_to tensorboard \
  --dataloader_num_workers 10 > ${output_path}/train.log 2>&1 &
wait