Network="Aquila2"

LOG=""

for para in $*; do
  if [[ $para == --log* ]]; then
    LOG=$(echo ${para#*=})
  fi
done

if [ ! -f "$LOG"]; then
  echo "log file is required!"
  exit 1
fi

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

output_path=${cur_path}/test/output/${ASCEND_DEVICE_ID}

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
AverageTime=`grep "elapsed time per iteration" $LOG | awk -F ':' '{print$3}' | awk -F '|' '{print$1}' | tail -n +21 | awk '{sum+=$1} END {print"",sum/NR/1000}'`
GBS=`grep "global-batch-size" $LOG | awk -F 'global-batch-size' '{print$2}' | awk -F ' ' '{print$1}'`
SEQ_LEN=`grep "seq-length" $LOG | awk -F 'seq-length' '{print$2}' | awk -F ' ' '{print$1}'`
NNODES=`grep "nnodes" $LOG | awk -F 'nnodes' '{print$2}' | awk -F ' ' '{print$1}'`
NPROC_PER_NODE=`grep "nproc-per-node" $LOG | awk -F 'nproc-per-node' '{print$2}' | awk -F ' ' '{print$1}'`
FPS=`echo "scale=2; $GBS * $SEQ_LEN / $AverageTime / $NNODES / $NPROC_PER_NODE" | bc`

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=`grep "lm loss:" $LOG | awk -F ':' '{print$3}' | awk -F '|' '{print$1}' | tail -n 1`

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${GBS}
DeviceType=$(uname -m)
DeviceNum=`echo $NNODES $NPROC_PER_NODE | awk '{printf"%d\n", $1*$2}'`
CaseName=${Network}_bs${BatchSize}_${DeviceNum}p_'acc'

#单迭代训练时长
TrainingTime=$AverageTime

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log