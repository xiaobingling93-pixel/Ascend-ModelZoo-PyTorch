#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
# 指定训练所使用的npu device卡id
device_id=0

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0


work_dir="output/fcn-1p-full/ckpt"
resume=0
performance=0

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="fcn_r18b"
#训练step
#train_steps=`expr 1281167 / ${batch_size}`
batch_size=2
num_workers=2
max_iters=80000
val_interval=8000

#TF2.X独有，需要模型审视修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path		           source data of training
    --performance              switch to performance mode when != 0
    --work_dir                 set output dir for training
    --resume                   resume training when != 0
    --batch_size               batch size for train dataloader
    --num_workers              num workers for train dataloader
    -h/--help		             show help message
    "
    exit 1
fi
#参数校验，不需要修改
#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --performance* ]];then
        performance=`echo ${para#*=}`
    elif [[ $para == --work_dir* ]];then
        work_dir=`echo ${para#*=}`
    elif [[ $para == --resume* ]];then
        resume=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --num_workers* ]];then
        num_workers=`echo ${para#*=}`
    fi
done

if (($performance!=0)); then
    max_iters=500
    val_interval=500
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

test_path_dir=$cur_path
ASCEND_DEVICE_ID=$device_id
export ASCEND_RT_VISIBLE_DEVICES=$ASCEND_DEVICE_ID
if [ ! -d ${test_path_dir}/output ];then
    mkdir ${test_path_dir}/output
fi
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
fi


#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/test/env_npu.sh
fi


#设置环境变量，不需要修改
echo "Device ID: $ASCEND_DEVICE_ID"
export RANK_ID=$RANK_ID
export WORLD_SIZE=1

if (($resume==0)); then
    python3 ./tools/train.py ./configs/fcn/fcn_r18b-d8_4xb2-80k_cityscapes-769x769.py \
    --work-dir=${work_dir} \
    --cfg-options train_cfg.max_iters=$max_iters \
    --cfg-options train_cfg.val_interval=$val_interval \
    --cfg-options train_dataloader.batch_size=$batch_size \
    --cfg-options train_dataloader.num_workers=$num_workers \
    --cfg-options data_root=$data_path \
    --cfg-options train_dataloader.dataset.data_root=$data_path \
    --cfg-options val_dataloader.dataset.data_root=$data_path \
    --cfg-options test_dataloader.dataset.data_root=$data_path
else
    python3 ./tools/train.py ./configs/fcn/fcn_r18b-d8_4xb2-80k_cityscapes-769x769.py \
    --work-dir=${work_dir} --resume \
    --cfg-options train_cfg.max_iters=$max_iters \
    --cfg-options train_cfg.val_interval=$val_interval \
    --cfg-options train_dataloader.batch_size=$batch_size \
    --cfg-options train_dataloader.num_workers=$num_workers \
    --cfg-options data_root=$data_path \
    --cfg-options train_dataloader.dataset.data_root=$data_path \
    --cfg-options val_dataloader.dataset.data_root=$data_path \
    --cfg-options test_dataloader.dataset.data_root=$data_path
fi


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

log_file=`find ${work_dir} -regex ".*\.log" | sort -r | head -n 1`

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'Iter(train)'  ${log_file}|awk -F " time: " '{print $NF}'|awk -F " " '{print $1}' | awk '{ sum += $0; n++} END { if (n > 0) print sum / n;}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改

train_accuracy=`grep -a 'mIoU' ${log_file}|awk 'END {print}'|awk -F "mIoU:" '{print $NF}'|awk -F " " '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'miou'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`
echo "TrainingTime for step(ms) : $TrainingTime"

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Iter(train)" ${log_file}|awk -F "loss:" '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "miou = ${train_accuracy}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
