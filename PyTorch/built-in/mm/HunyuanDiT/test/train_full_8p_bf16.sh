# 网络名称，权重路径以及相关参数，需要模型审视修改
Network="HunyuanDiT"
BATCH_SIZE=1
max_train_steps=5000
task_flag="dit_g2_full_1024p"                                # the task flag is used to identify folders.
resume=./ckpts/t2i/model/                                    # checkpoint root for resume
index_file=dataset/porcelain/jsons/porcelain_mt.json         # index file for dataloader
results_dir=./log_EXP                                        # save root for results
image_size=1024                                              # training image resolution
grad_accu_steps=1                                            # gradient accumulation
warmup_num_steps=0                                           # warm-up steps
lr=0.0001                                                    # learning rate
ckpt_every=10000                                             # create a ckpt every a few steps.
ckpt_latest_every=5000                                       # create a ckpt named `latest.pt` every a few steps.

export WORLD_SIZE=8
export MASTER_PORT=29500
export MASTER_ADDR=127.0.0.1

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

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/${ASCEND_DEVICE_ID}

mkdir -p ${output_path}

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
deepspeed --num_gpus ${WORLD_SIZE} --num_nodes 1 --master_port=${MASTER_PORT} hydit/train_deepspeed.py ${params} \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.03 \
    --predict-type v_prediction \
    --uncond-p 0.44 \
    --uncond-p-t5 0.44 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${BATCH_SIZE} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --use-ema \
    --ema-dtype fp32 \
    --results-dir ${results_dir} \
    --resume-split \
    --resume ${resume} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 1 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --multireso \
    --reso-step 64 \
    --epochs 1400 \
    --max-training-steps ${max_train_steps} \
    --norm "layer"  \
    --autocast-dtype  "bf16" \
 > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))


# 训练用例信息，不需要修改
BatchSize=${BATCH_SIZE}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
avg_time=$(grep -a 'Steps/Sec:' "${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log" |
           awk -F "Steps/Sec:" '{print $2}' |
           awk '{a+=$1} END {if (NR!=0) printf "%.3f\n", a/NR}')
FPS=$(echo "$avg_time * $BatchSize" |bc)
# 输出性能100步到200步平均单步耗时
avg_millisec_per_step=$(grep -a 'step=00001[0-9][0-9]' "${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log" |
           awk -F "Millisec/Step:" '{print $2}' |
           awk '{a+=$1} END {if (NR!=0) printf "%.3f\n", a/NR}')
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"
echo "avg_millisec_per_step(100-200step) : $avg_millisec_per_step"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${BATCH_SIZE}'*8/'${FPS}'}')


# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "AvgTrainingTime = ${avg_millisec_per_step}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_avg_millisec_per_step.log