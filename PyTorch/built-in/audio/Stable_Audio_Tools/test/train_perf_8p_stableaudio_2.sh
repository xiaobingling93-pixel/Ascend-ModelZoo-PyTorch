# 微调生成的ckpt路径
Network="stableaudio2"


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
mkdir -p ${test_path_dir}/output/${ASCEND_DEVICE_ID}


#推理开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

python ./train.py \
 --config-file ./perf.ini \
 --dataset-config ./stable_audio_tools/configs/dataset_configs/local_training.json \
 --model-config  ./stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json \
 --name stableaudio2_perf \
 > ${test_path_dir}/output/$ASCEND_DEVICE_ID/perf_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))


# 训练用例信息，不需要修改
DeviceType=$(uname -m)
CaseName=${Network}_${WORLD_SIZE}'p'_'perf'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'train time:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/perf_${ASCEND_DEVICE_ID}.log|awk -F "train time: " '{print $2}' | tail -100 | awk '{a+=$1} END {if (NR != 0) printf("%.2f",a/NR)}'`
#loss值
ActualLoss=$(grep -o "train/loss=[0-9.]*" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/perf_${ASCEND_DEVICE_ID}.log | awk 'END {print $NF}')

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
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log