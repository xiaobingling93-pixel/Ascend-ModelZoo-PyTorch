# FlashVSR-v1.1-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [获取权重](#获取权重)
    - [环境变量](#环境变量)
    - [执行推理](#执行推理)
    - [性能数据](#性能数据)
## 概述
FlashVSR 是一个面向实时扩散模型的流式视频超分 辨率框架，旨在通过实现高效性、可扩展性和实时性能，使基于扩散模型的视频超分辨率技术变得实用化。

本文的介绍了FlashVSR-v1.1模型的部署流程，包括推理环境准备、模型部署、功能验证，旨在帮助用户快速完成模型部署和验证。

- 版本说明：


```
url=https://github.com/OpenImagingLab/FlashVSR
commit_id=b527c6f285fb30df530f5febc8b45764a789c961
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
cd ModelZoo-PyTorch/MindIE/MultiModal/FlashVSR-v1.1
git clone https://github.com/OpenImagingLab/FlashVSR.git
cd FlashVSR
git reset --hard b527c6f285fb30df530f5febc8b45764a789c961
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

cd FlashVSR
pip install -e .
pip install -r requirements.txt
cd ../MindIE-SD
python setup.py bdist_wheel
cd ..
pip install MindIE-SD/dist/xxx.whl #根据实际情况修改xxx
```
### 获取权重
```
cd FlashVSR/examples/WanVSR
git lfs install
git clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
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
```
# full模式
python infer_flashvsr_v1.1_full.py
# tiny模式
python infer_flashvsr_v1.1_tiny.py
# tiny_long模式
python infer_flashvsr_v1.1_tiny_long_video.py
```
具体推理文件以及结果保存路径可以在推理脚本中配置，默认从`inputs`文件夹中读取视频文件(warm_up和正式推理文件)，输出结果到`results`中
### 性能数据
注：为了获取真实性能数据，推理前需要先进行warm_up  
|机器|模式|输入尺寸|时长|放大倍率|输出|推理时长|
|--|--|--|--|--|--|--|
|Atlas 800T A2|full|384x384@30fps|2s|2.0|768x768@30fps|13.85s|
|Atlas 800T A2|full|672x384@30fps|3s|2.0|1280x768@30fps|25.60s|
|Atlas 800T A2|full|384x672@30fps|3s|2.0|768x1280@30fps|23.31s|
|Atlas 800T A2|full|640x480@30fps|2s|2.0|1280x896@30fps|23.35s|
|Atlas 800T A2|tiny|384x384@30fps|2s|2.0|768x768@30fps|11.25s|
|Atlas 800T A2|tiny|672x384@30fps|3s|2.0|1280x768@30fps|20.20s|
|Atlas 800T A2|tiny|384x672@30fps|3s|2.0|768x1280@30fps|19.99s|
|Atlas 800T A2|tiny|640x480@30fps|2s|2.0|1280x896@30fps|19.04s|
|Atlas 800T A2|tiny-long|768x416@30fps|54s|2.0|1536x768@30fps|498.74s|