#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="OpenFold_for_PyTorch"

cur_path=`pwd`
max_epochs=1

for para in $*
do
   if [[ $para == --max_epochs* ]];then
      	max_epochs=`echo ${para#*=}`
   elif [[ $para == --data_path* ]];then
       data_path=`echo ${para#*=}`
   elif [[ $para == --val_alignment_dir* ]];then
       val_alignment_dir=`echo ${para#*=}`
   elif [[ $para == --val_data_dir* ]];then
       val_data_dir=`echo ${para#*=}`
   fi
done

source ${cur_path}/test/env_npu.sh

python3 train_openfold.py  $data_path/pdb_data/mmcif_files \
                           $data_path/alignment_data/alignment_dbs \
                           $data_path/pdb_data/mmcif_files \
                           ./output \
                           "2021-10-10" \
                           --train_chain_data_cache_path $data_path/pdb_data/data_caches/chain_data_cache.json \
                           --template_release_dates_cache_path $data_path/pdb_data/data_caches/mmcif_cache.json \
                           --alignment_index_path $data_path/alignment_data/alignment_dbs/alignment_db.index \
                           --config_preset initial_training \
                           --seed 42 \
                           --obsolete_pdbs_file_path $data_path/pdb_data/obsolete.dat \
                           --num_nodes 1 \
                           --gpus 8 \
                           --max_epochs $max_epochs \
                           --deepspeed_config_path ./deepspeed_config.json \
                           --resume_from_ckpt ./output/checkpoints/0-1250.ckpt \
                           --val_data_dir $val_data_dir \
                           --val_alignment_dir $val_alignment_dir \
                           --openfold_run_stage val \
                           > openfold_val_8p.log 2>&1 &

wait

# 提取最后一行中的 val/loss 小数值并赋值给变量
val_loss=$(grep 'val/loss' openfold_val_8p.log | tail -n1 | awk '{for(i=1;i<NF;i++) if($i ~ /^[0-9]+(\.[0-9]+)?$/) print $i}')

# 输出结果
echo "val/loss: $val_loss"