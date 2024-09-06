#!/bin/bash

# 基础参数
Network="InternVL1.5"
BATCH_SIZE=32
PER_DEVICE_BATCH_SIZE=1
max_train_steps=5000

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        BATCH_SIZE=$(echo ${para#*=})
    elif [[ $para == --max_train_steps* ]]; then
        max_train_steps=$(echo ${para#*=})
    fi
done

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

source ${test_path_dir}/env_npu.sh

# 创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output

mkdir -p ${output_path}

# 配置训练参数和环境变量
WORLD_SIZE=`echo ${test_path_dir}/hostfile | wc -l`
GRADIENT_ACC=$((BATCH_SIZE / WORLD_SIZE / PER_DEVICE_BATCH_SIZE))
export DS_ENV_FILE=${test_path_dir}/deepspeed_env

# 开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

cd internvl_chat && \
deepspeed \
  --hostfile ${test_path_dir}/hostfile \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "./pretrained/InternVL-Chat-V1-5" \
  --conv_style "internlm2-chat" \
  --output_dir ${output_path} \
  --meta_path "./shell/data/internvl_1_2_finetune.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --max_steps ${max_train_steps} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 9999 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard" \
  > ${output_path}/train.log 2>&1 &
  
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))


# 训练用例信息，不需要修改
BatchSize=${BATCH_SIZE}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'train_samples_per_second =' ${output_path}/train.log | awk -F "train_samples_per_second =" '{print $2}' | sed 's/[[:space:]]//g'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"


# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${BATCH_SIZE}'*8/'${FPS}'}')

# 从train.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep {\'loss\': ${output_path}/train.log | awk -F "{'loss': " '{print $NF}' | awk -F "," '{print $1}' > ${output_path}/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`grep -a 'train_loss' ${output_path}/train.log | awk 'NR==2 {print}' |awk -F "=" '{print $2}' | sed 's/[[:space:]]//g'`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${CaseName}_perf_report.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}_perf_report.log
