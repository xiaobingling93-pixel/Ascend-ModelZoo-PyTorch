 #!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="SSD-MobileNetV1"
# 训练batch_size
batch_size=32
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""
# 测试集路径
validation_data_path=""
# 训练epoch
train_epochs=240
# 指定训练所使用的npu device卡id
device_id=0
# 加载数据进程数
workers=32
# 设置是否执行评测的变量，1为执行，0则不执行
EXEC_EVAL=1


# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --validation_data_path* ]];then
        validation_data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
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


#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
echo ${data_path}
echo ${validation_data_path}
setsid python3 ./train_ssd.py \
    --datasets=${data_path} \
    --validation_dataset=${validation_data_path} \
    --num_workers=${workers} \
    --lr=0.01 \
    --base_net models/mobilenet_v1_with_relu_69_5.pth \
    --num_epochs=${train_epochs} \
    --t_max=240 \
    --scheduler=cosine \
    --amp=True \
    --debug_steps=1 \
    --local_rank=${device_id} \
    --main_rank=${device_id} \
    --batch_size=${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| tail -n +2 |awk '{a+=$19}END{print a/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出验证损失,需要模型审视修改
validation_loss=`grep -a 'Validation'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Validation" '{print $2}'|awk -F " " '{print $2}' | awk -F ',' '{print $1}'`
#打印，不需要修改
echo "Final Validation Loss : ${validation_loss}"
echo "E2E Training Duration sec : $e2e_time"
echo "Batch Size: ${batch_size}"
echo "Epoch: ${train_epochs}"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "Epoch = ${train_epochs}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ValidationLoss = ${validation_loss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

#################启动评测脚本#################
# 如果 EXEC_EVAL 为 1，则执行评测
if [ "$EXEC_EVAL" -eq 1 ]; then
    # 计算 epoch - 1
    eva_epoch_index=$((train_epochs - 1))

    # 查找 models 目录下包含 "$eva_epoch_index" 且以 .pth 结尾的文件，并按生成时间排序，取最新的文件
    model_file=$(find models -type f -name "*Epoch-${eva_epoch_index}*.pth" -printf "%T@ %p\n" | sort -n -r | head -n 1 | awk '{print $2}')

    # 判断是否找到匹配的文件
    if [ -z "$model_file" ]; then
        echo "No matching model file is found."
        exit 1
    fi

    # 打印找到的模型文件
    echo "Model File Found: $model_file"

    # 传入评测命令
    eval_command="bash ./test/train_eval.sh --data_path=${validation_data_path} --pth_path=$model_file"
    echo "Executing eva command: $eval_command"
    $eval_command
else
    echo "Eval did not execute because the EXEC_EVAL variable is set to $EXEC_EVAL."
fi
mAP=`awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log`

# 补充关键信息打印到${CaseName}.log中
echo ${mAP} >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
