# 微调生成的ckpt路径
Network="OpenSora"
BATCH_SIZE=1
max_train_steps=0
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

ASCEND_DEVICE_ID=0
#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}
else
    mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}
fi

#推理开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

torchrun --nnodes=1 --nproc_per_node=${WORLD_SIZE} --master-port ${MASTER_PORT} scripts/train.py \
 configs/opensora/train/120x256x256.py \
 --batch-size ${BATCH_SIZE} \
 --max-train-steps ${max_train_steps} \
 >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &

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
FPS=`grep -a 'FPS' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS " '{print $2}' | tail -100 | awk '{a+=$1} END {if (NR != 0) printf("%.2f",a/NR)}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"


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