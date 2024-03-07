# 网络名称,同目录名称,需要模型审视修改
Network="StableDiffusionXLMT5"

# 预训练模型
pretrain_model_path="stable-diffusion-xl-base-1.0"
vae_model_path="sdxl-vae-fp16-fix"
dataset_name="laion"
mt5_tokenizer_path="Taiyi-Stable-Diffusion-1B-Chinese-v0.1/tokenizer/"
mt5_encoder_path="mt5-xxl/"
batch_size=2
max_train_epoch=12
mixed_precision="bf16"
resolution="1024,1024"


for para in $*; do
  if [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_epoch* ]]; then
    max_train_epoch=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
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

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"


python3 -m torch.distributed.launch --nproc_per_node 8 --use_env \
  ./sdxl_pretrain_mt5.py \
  --pretrained_model_name_or_path=${pretrain_model_path} \
  --vae=$vae_model_path \
  --mt5_tokenizer_path=$mt5_tokenizer_path \
  --mt5_encoder_path=$mt5_encoder_path \
  --train_data_dir=${dataset_name} \
  --resolution=${resolution} \
  --enable_bucket \
  --max_data_loader_n_workers=8 \
  --min_bucket_reso=512 \
  --max_bucket_reso=2048 \
  --output_dir=${output_path} \
  --output_name="test_sdxl" \
  --logging_dir="./logs" \
  --max_train_epoch=${max_train_epoch} \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --lr_scheduler="constant" \
  --min_snr_gamma=5 \
  --learning_rate=1e-5 \
  --learning_rate_te1=1e-7 \
  --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --lr_scheduler_num_cycles=1 \
  --train_batch_size=${batch_size} \
  --mixed_precision=${mixed_precision} \
  --seed=1 \
  --caption_extension=".txt" \
  --keep_tokens=0 \
  --optimizer_type="oss" \
  --max_token_length=512 \
  --no_half_vae \
  --enable_npu_flash_attention \
  --mt5_dim_size=4096 \
  --bucket_reso_steps=64 > ${output_path}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait
 
 #训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${ASCEND_DEVICE_ID}.log | awk '{print $NF}' | sed -n '1000,1999p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "step_loss=[0-9.]*" ${output_path}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*8/'${FPS}'}')

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log
