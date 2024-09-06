# export CUDA_VISIBLE_DEVICES=2,6,7
source /path_to_cann/set_env.sh
export OPENAI_API_KEY=...
num_frames=16
test_ratio=1

# 13b, uses offload thus saving the full model
model_dir=/path_to_model
weight_dir=/path_to_train_result
SAVE_DIR=test_results/test_pllava_13b
lora_alpha=4
video_path=/path_to_PLLaVA/example/cooking.mp4

conv_mode=eval_videoqabench
python -m tasks.eval.videoqabench.pllava_eval_single \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/videoqabench \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --test_ratio ${test_ratio} \
    --example_path ${video_path} \
    --eval_mode 1
