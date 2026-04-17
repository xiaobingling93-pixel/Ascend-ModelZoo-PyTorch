# FlashVSR-Pro-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [获取权重](#获取权重)
    - [环境变量](#环境变量)
    - [执行推理](#执行推理)
    - [性能数据](#性能数据)
## 概述
FlashVSR-Pro 是FlashVSR的非官方增强版模型，可以直接用于生产环境当作。

本文的介绍了FlashVSR-Pro模型的部署流程，包括推理环境准备、模型部署、功能验证，旨在帮助用户快速完成模型部署和验证。

- 版本说明：


```
url=https://github.com/LujiaJin/FlashVSR-Pro/
commit_id=bf31fdd4a21642ba668ef94065aa3986608ceb1f
```

## 推理环境准备
- 该模型需要以下插件与驱动  
**表 1** 版本配套表

|配套|版本|环境准备指导|
|--|--|--|
|固件与驱动|25.5.2| [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)|
|CANN|8.5.0|-|
|Python|3.11|-|
|PyTorch|2.6.0|-|
|Ascend Extension PyTorch|2.6.0.post5|-|
|硬件|Atlas 800T A2, Atlas 800I A2|-|

## 快速上手
### 获取源码
1. 获取`PyTotch`源码
```
git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/MindIE/MultiModal/FlashVSR-Pro
git clone https://github.com/LujiaJin/FlashVSR-Pro.git
cd FlashVSR-Pro
git reset --hard bf31fdd4a21642ba668ef94065aa3986608ceb1f
cd ..
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
git reset --hard 4aa3014c21ea171c3255d2d2591debeaac9e5202
cd ..
```
2. 修改第三方库
```
# 注：patch命令只能执行一次，第二次执行会报错
cd FlashVSR
git apply ../diff.patch
cd ..
```
3. 安装依赖
```
# openeuler
yum update
yum -y install opencv ffmpeg
# ubuntu
apt-get update
apt-get install -y libgl1 libglib2.0-0 ffmpeg

cd FlashVSR-Pro
pip install -e .
pip install -r requirements.txt
cd ../MindIE-SD
python setup.py bdist_wheel
cd ..
pip install MindIE-SD/dist/xxx.whl #根据实际情况修改xxx
```
### 获取权重
```
cd FlashVSR-Pro
git lfs install
git clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 ./models/FlashVSR-v1.1
```
### 环境变量
```
# 一级流水优化
export TASK_QUEUE_ENABLE=1
# combind标志，用于优化两个非连续算子组合类场景
export COMBIND_ENABLE=1
# CPU绑核
export CPU_AFFINITY_CONF=1
```
### 执行推理
运行推理脚本
```
python infer.py \
-i ./inputs/example0.mp4 \
-o ./results/ \
--mode tiny \
--tile-dit \
--tile-vae \
--tile-size 256 \
--overlap 24 \
--keep-audio \
--scale 2.0 \
--seed 0 \
--sparse-ratio 2.0 \
--kv-ratio 3.0 \
--local-range 11 \
--color-fix \
--fps 30 \
--quality 10 \
--warmup_file ./inputs/example0.mp4 \
--device cuda \
--dtype bf16
```
参数说明
i：输入视频文件路径。  
o：输出视频文件路径。  
mode：推理模式，支持full（使用Wan VAE）、tiny（使用TCDecoder）和tiny-long（适用于长视频）。  
tile-dit：开启dit分片推理（减少显存占用）。  
tile-vae：开启vae分片decode（仅full模式生效）。  
tile-size：分片大小。  
overlap：分片之间重叠区域大小。  
keep-audio：保留原视频中的音频（如果有的话，且生成视频的fps需要与原视频保持一致）。  
scale：视频缩放倍率。  
seed：随机种子。  
sparse-ratio：稀疏attention比率（1.5=更快，2=更稳定）。  
kv-ratio：KV缓存率。  
local-range：局部注意力范围（9=更敏锐，11=更稳定）。  
color-fix：开启色彩校准。  
fps：生成视频fps。  
quality：生成视频质量（0-10）。  
warmup_file：warm up视频文件。  
device：使用的设备（cuda/cpu）。  
dtype：数据类型，支持fp32，fp16，bf16。  


命令样例
```
python infer.py \
-i ./inputs/example0.mp4 \
-o ./results/ \
--mode tiny \
--warmup_file ./inputs/example0.mp4
```
### 性能数据
注：为了获取真实性能数据，推理前需要先进行warm_up  
|机器|模式|输入尺寸|时长|放大倍率|输出|推理时长|
|--|--|--|--|--|--|--|
|Atlas 800T A2|full|384x384@30fps|2s|2.0|768x768@30fps|22.27s|
|Atlas 800T A2|full|672x384@30fps|3s|2.0|1280x768@30fps|56.43s|
|Atlas 800T A2|full|384x672@30fps|3s|2.0|768x1280@30fps|43.07s|
|Atlas 800T A2|full|640x480@30fps|2s|2.0|1280x896@30fps|60.57s|
|Atlas 800T A2|tiny|384x384@30fps|2s|2.0|768x768@30fps|11.36s|
|Atlas 800T A2|tiny|672x384@30fps|3s|2.0|1280x768@30fps|22.01s|
|Atlas 800T A2|tiny|384x672@30fps|3s|2.0|768x1280@30fps|22.18s|
|Atlas 800T A2|tiny|640x480@30fps|2s|2.0|1280x896@30fps|21.86s|
|Atlas 800T A2|tiny-long|768x416@30fps|54s|2.0|1536x768@30fps|553.73s|