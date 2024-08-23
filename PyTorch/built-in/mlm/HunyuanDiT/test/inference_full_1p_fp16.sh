# 网络名称，权重路径以及相关参数，需要模型审视修改
Network="HunyuanDiT"
prompt="渔舟唱晚"
image_size_height=1280
image_size_weight=768

for para in $*
do
    if [[ $para == --prompt* ]]; then
        prompt=$(echo ${para#*=})
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

#推理开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

python sample_t2i.py --infer-mode fa --prompt ${prompt} --image-size ${image_size_height} ${image_size_weight} --no-enhance \
         > ${test_path_dir}/output/$ASCEND_DEVICE_ID/inference_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#推理结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))


# 训练用例信息，不需要修改
BatchSize=${BATCH_SIZE}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
