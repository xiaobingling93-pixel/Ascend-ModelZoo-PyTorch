# 网络名称,同目录名称,需要模型审视修改
Network="Aquila2"

# 预训练模型
CONFIG="./ascend/scripts/pretrain_aquila_70B_distributed.json"
EXTRA_CONFIG=""
ACTION="run"
MODE=""
HOSTFILE=""

for para in $*; do
  if [[ $para == --config* ]]; then
    CONFIG=$(echo ${para#*=})
  elif [[ $para == --extra-config* ]]; then
    EXTRA_CONFIG=$(echo ${para#*=})
  elif [[ $para == --action* ]]; then
    ACTION=$(echo ${para#*=})
  elif [[ $para == --mode* ]]; then
    MODE=$(echo ${para#*=})
  elif [[ $para == --hostfile* ]]; then
    HOSTFILE=$(echo ${para#*=})
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

echo ${test_path_dir}

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/${ASCEND_DEVICE_ID}

if [ -d ${output_path} ]; then
  rm -rf ${output_path}
fi

mkdir -p ${output_path}

if [[ $MODE == "performance"]]; then
  GBS=$(grep "global_batch_size" $EXTRA_CONFIG | awk '{print $NF}')
  SAMPLES=$(echo $GBS | awk '{printf"%d\n", $1*100}')
  cp $EXTRA_CONFIG ${output_path}/tmp.json
  EXTRA_CONFIG=${output_path}/tmp.json
  sed -i "s/\"train_samples\":.*$/\"train_samples\": $SAMPLES,/" ${output_path}/tmp.json
fi

if [ ! -f "$HOSTFILE"]; then
  echo "hostfile is required!"
  exit 1
else 
  if [ ! -f "${output_path}/tmp.json" ]; then
    cp $EXTRA_CONFIG ${output_path}/tmp.json
  fi
  HOSTFILE=$(echo $HOSTFILE | sed 's/[\/]/\\&/g')
  sed -i "s/\"hostfile\":.*$/\"hostfile\": \"$HOSTFILE\",/" ${output_path}/tmp.json
fi

python3 run.py \
 --config=$CONFIG \
 --extra-config=$EXTRA_CONFIG \
 --action=$ACTION > ${output_path}/train_70B.log 2>&1 &
wait

if [ -f "${output_path}/tmp.json" ]; then
  rm -rf ${output_path}/tmp.json
fi