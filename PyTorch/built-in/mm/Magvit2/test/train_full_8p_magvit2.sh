# 网络名称,同目录名称,需要模型审视修改
Network="Magvit2"
dataset_folder=""
batch_size=16
grad_accum_every=1
learning_rate=2e-5
num_train_steps=5000

for para in $*; do
if [[ $para == --dataset_folder* ]]; then
    dataset_folder=$(echo ${para#*=})
  elif [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --grad_accum_every* ]]; then
    grad_accum_every=$(echo ${para#*=})
  elif [[ $para == --learning_rate* ]]; then
    learning_rate=$(echo ${para#*=})
  elif [[ $para == --num_train_steps* ]]; then
    num_train_steps=$(echo ${para#*=})
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
mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}

num_processes=8
train_script=train.py

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

accelerate launch \
 --num_processes=$num_processes \
 --mixed_precision=bf16 \
 --main_process_port=1830 \
 $train_script \
 --dataset_folder=$dataset_folder \
 --batch_size=$batch_size \
 --grad_accum_every=$grad_accum_every \
 --learning_rate=$learning_rate \
 --num_train_steps=$num_train_steps > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))


# 训练用例信息，不需要修改
DeviceType=$(uname -m)
CaseName=${Network}_bs${batch_size}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'step_total_time: ' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "step_total_time: " '{print $2}' | tail -20 | awk '{a+=$1} END {if (NR != 0) printf("%.2f",a/NR)}'`
#loss值
ActualLoss=$(grep -o "recon loss: [0-9.]*" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 获取性能数据，不需要修改

ActualFPS=${FPS}
echo "Final Performance : $ActualFPS s/step"


# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log