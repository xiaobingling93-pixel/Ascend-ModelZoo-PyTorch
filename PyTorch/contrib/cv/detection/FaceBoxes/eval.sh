train_epoch=300
currentDir=`pwd`
cp eval_$train_epoch/FDDB_dets.txt FDDB_Evaluation/
cd FDDB_Evaluation
# 如果存在缓存数据文件，则删除目录
rm -rf convert
rm -rf compare_data
python3 convert.py
python3 split.py
python3 evaluate.py
