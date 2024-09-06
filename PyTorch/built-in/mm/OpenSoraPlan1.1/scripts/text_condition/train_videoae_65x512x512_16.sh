#!/bin/bash

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

if [ ! -d $test_path_dir ]; then
    mkdir -p $test_path_dir
fi

log_file=$test_path_dir/train_$(date +%y%m%d%H%M).log

start_time=$(date +%s)

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "LanguageBind/Open-Sora-Plan-v1.1.0/vae" \
    --video_data "scripts/train_data/video_data.txt" \
    --use_img_from_vid \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode math \
    --train_batch_size=2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="tensorboard" \
    --checkpointing_steps=500 \
    --output_dir="65x512x512_10node_bs2_lr2e-5_4img" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 16 \
    --enable_tiling \
    --pretrained LanguageBind/Open-Sora-Plan-v1.1.0/65x512x512/diffusion_pytorch_model.safetensors > $log_file 2>&1 &
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
avg_time=`grep -a 'step: '  $log_file|awk -F "total time:" '{print $2}' | awk -F ", " '{print $1}' | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
Iteration_time=$avg_time

# 打印，不需要修改
echo "Iteration time : $Iteration_time"

# 打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"
