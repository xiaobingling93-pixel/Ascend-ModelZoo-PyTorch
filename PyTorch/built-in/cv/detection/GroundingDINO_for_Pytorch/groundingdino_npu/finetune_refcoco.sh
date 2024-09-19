#!/bin/bash
Network="GroundingDINO"

source groundingdino_npu/env_npu.sh
PYTHON_PATH="Python Env Path"
export Mx_Driving_PYTHON_PATH=${PYTHON_PATH}/lib/python3.8
export ASCEND_CUSTOM_OPP_PATH=${Mx_Driving_PYTHON_PATH}/site-packages/mx_driving/packages/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

CONFIG=groundingdino_npu/mm_grounding_dino_swin-b_finetune_b2_refcoco.py
GPUS=8
NNODES=1
batch_size=2
NODE_RANK=0
PORT=6005
MASTER_ADDR=localhost

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

# 创建测试日志输出路径
if [ -d ${cur_path}/test/output ]; then
  rm -rf ${cur_path}/test/output
  mkdir -p ${cur_path}/test/output
else
  mkdir -p ${cur_path}/test/output
fi


#训练开始时间，不需要修改
start_time=$(date +%s)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_npu.py \
    $CONFIG \
    --launcher pytorch ${@:3} > ./train.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${GPUS}'p'_'acc'

#结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
ava_time=$(grep "time: " ./train.log | awk '{print $18}' | sed -n '3,200p' | awk '{a+=$1}END{print a/NR}')
echo "avarage time: $ava_time"
GPU_COMPARE=$(awk 'BEGIN{printf "%.2f\n", 1.0607/'${ava_time}'}')
echo "compare with A100: $GPU_COMPARE * A100"
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*'${GPUS}'/'${ava_time}'}')
echo "Final Performance images/sec : $ActualFPS"

# 输出训练精度,需要模型审视修改
mean_precision=$(grep "refcoco_val/refexp/mean_precision:" ./train.log | tail -n 1| awk -F "refcoco_val/refexp/mean_precision: " '{print $2}' |awk -F ' ' '{print $1}')
precision_at_1=$(grep "refcoco_val/refexp/refcoco_precision@1" ./train.log | tail -n 1 | awk -F "refcoco_precision@1: " '{print $2}'|awk -F ' ' '{print $1}')
precision_at_5=$(grep "refcoco_val/refexp/refcoco_precision@5" ./train.log | tail -n 1 | awk -F "refcoco_precision@5: " '{print $2}'|awk -F ' ' '{print $1}')
precision_at_10=$(grep "refcoco_val/refexp/refcoco_precision@10" ./train.log | tail -n 1 | awk -F "refcoco_precision@10: " '{print $2}'|awk -F ' ' '{print $1}')
echo "mean precision: $mean_precision"
echo "precision@1: $precision_at_1"
echo "precision@5: $precision_at_5"
echo "precision@10: $precision_at_10"
echo "E2E Training Duration sec : $e2e_time"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >>${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${GPUS}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${batch_size}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "ava_time = ${ava_time}" >>${test_path_dir}/output/${CaseName}.log
echo "GPU_COMPARE = ${GPU_COMPARE}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${CaseName}.log
echo "mean precision = ${mean_precision}" >>${test_path_dir}/output/${CaseName}.log
echo "precision@1 = ${precision_at_1}" >>${test_path_dir}/output/${CaseName}.log
echo "precision@5 = ${precision_at_5}" >>${test_path_dir}/output/${CaseName}.log
echo "precision@10 = ${precision_at_10}" >>${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log