# ------------------------------------------------------------------------
# Copyright 2024 Huawei Technologies Co., Ltd
#-------------------------------------------------------------------------

#!/bin/bash


#集合通信参数,不需要修改
export RANK_SIZE=8
RANK_ID_START=0

#网络名称,同目录名称,需要模型审视修改
Network="PETR_for_Pytorch"

#训练epoch
train_epochs=24

#训练batch_size,,需要模型审视修改
batch_size=($RANK_SIZE * 1)
echo ${batch_size}

#训练配置文件
export CONFIG="projects/configs/petr/petr_r50dcn_gridmask_p4.py"

# checkpoint文件
CHECKPOINT="work_dirs/petr_r50dcn_gridmask_p4/latest.pth"

#petr训练监听端口
export PORT=${PORT:-28500}

# petr 精度监听端口
export PORT_EVAL=${PORT:-29500}
###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#创建DeviceID输出目录，不需要修改
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi  

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本，需要模型审视修改
echo "------------------ Start train please wait ------------------"
cd $cur_path/
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python3 -m torch.distributed.launch --nproc_per_node=${RANK_SIZE} --use_env --master_port=$PORT \
   $(dirname "$0")/../tools/train.py $CONFIG --launcher pytorch ${@:3} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    
wait

echo "------------------ Start eval please wait ------------------"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python3 -m torch.distributed.launch --nproc_per_node=${RANK_SIZE} --use_env --master_port=${PORT_EVAL} \
   $(dirname "$0")/../tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval bbox > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/result.log 2>&1 &

wait
#训练结束时间,不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


##################获取训练数据################
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_time=`grep -a 'time:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'NR==FNR{a[NR] = $0;next} END{print a[NR -1]}' | awk -F'time:' '{print $2}' | awk -F'[ ,]' '{print $2}'`
FPS=$(echo "scale=2; $batch_size/$step_time" | bc)
FPS=$(awk 'BEGIN{printf "%.2f\n",'$batch_size'/'$step_time'}')
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改

echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
#TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

###打印精度数据，并打印到${CaseName}.log中不需要修改
metric_names=("mAP" "mATE" "mASE" "mAOE" "mAVE" "mAAE" "NDS")

for metric_name in "${metric_names[@]}"; do
    metric_value=$(grep -o "${metric_name}:\s*[0-9]\+\(\.[0-9]\+\)\?\s*" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/result.log | awk -F': ' '{print $2}')
    echo "${metric_name}: ${metric_value}"
    echo "${metric_name} = ${metric_value}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
done

##关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log