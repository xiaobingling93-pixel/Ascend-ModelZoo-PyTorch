source /path_to_cann/set_env.sh
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

OUTPUT_DIR=./pllava_video_outputs/test_train_7b_reconstruct

pooling_shape=(16,12,12)

repo_id=/path_to_model/llava-hf/llava-v1.6-vicuna-7b-hf
accelerate launch --main_process_port 6876 --config_file scripts/accel_config_multigpu.yaml tasks/train/train_pllava_nframe_accel.py  \
    tasks/train/config_pllava_nframe.py \
    output_dir ${OUTPUT_DIR} \
    train_corpus videochat2_instruction_debug \
    save_steps 10000 \
    ckpt_epochs 100 \
    num_workers 8 \
    num_frames 16 \
    model.pooling_method avg \
    model.use_lora True \
    model.repo_id $repo_id \
    model.pooling_shape $pooling_shape \
    optimizer.lr 2e-5 \
    scheduler.epochs 1 \
    scheduler.warmup_ratio 0.2 \
    scheduler.min_lr_multi 0.25 \
    scheduler.is_videochat2_custom True \
    preprocess.mm_alone False \
    preprocess.random_shuffle False \
    preprocess.add_second_msg False



