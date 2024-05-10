# 网络名称,同目录名称,需要模型审视修改
Network="OpenSora"
data_path=""

for para in $*; do
  if [[ $para == --data_path* ]]; then
    data_path=$(echo ${para#*=})
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

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

torchrun --nnodes=1 --nproc_per_node=1 --master-port 61888 scripts/train.py \
configs/opensora-v1-1/train/stage1.py \
--data-path ${data_path} >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/infer_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))


# 训练用例信息，不需要修改
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
echo "E2E Training Duration sec : $e2e_time"


# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log