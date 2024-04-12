BatchSize=16
NGPUS=1
Network="bisenetv2"
CaseName=${Network}_bs${BatchSize}_${NGPUS}p_performance

mkdir -p test/output

start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source test/env_npu.sh
fi

export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config configs/bisenetv2_city.py --mode performance >test/output/train.log 2>&1

end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"
FPS=`grep -a 'iter:' test/output/train.log | sed '1,2d' | grep -oP 'time: \K[\d.]+' | awk -v bs="$BatchSize" -v cards="$NGPUS" '{a+=$1} END {printf("%.3f", bs*cards*100*NR/a)}'`
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

TrainingTime=`grep -a 'iter:' test/output/train.log | grep -oP 'time: \K[\d.]+' | awk  '{a+=$1} END {printf("%.3f", a)}'`
grep "iter:" test/output/train.log | grep -oP 'loss: \K[\d.]+' >test/output/train_${CaseName}_loss.txt

echo "Network = ${Network}" >test/output/${CaseName}.log
echo "RankSize = ${NGPUS}" >>test/output/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>test/output/${CaseName}.log
echo "DeviceType = $(uname -m)" >>test/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>test/output/${CaseName}.log

echo "ActualFPS = ${FPS}" >test/output/${CaseName}_perf_report.log
echo "TrainingTime = ${TrainingTime}" >>test/output/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>test/output/${CaseName}_perf_report.log