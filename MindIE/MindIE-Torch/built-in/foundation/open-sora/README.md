# OpenSora模型-推理指导

# 概述

Open Sora采用动态掩码策略等技术细节复现Sora，并已实现可变长宽比、可变分辨率和可变时长等功能。

- 设备支持：
Atlas 800I A2推理设备：支持的卡数为1
Atlas 300I Duo推理卡：支持的卡数为1

## 输入输出数据

输入一个prompt，输入一个2s长的视频

# 推理环境准备

**表 1**  版本配套表

| 配套                                 | 版本    | 环境准备指导 |
| ------------------------------------ | ------- | ------------ |
| Python                               | 3.10.13 | -            |
| PyTorch                              | 2.1.0   | -            |
| 硬件：Atlas 300I Duo ，Atlas 800I A2 | \       | \            |

请以CANN版本选择对应的固件与驱动版本。

# 快速上手

## 获取源码

1. 安装依赖

   ```bash
   pip3 install -r requirements.txt
   ```

2. 安装mindie包

   ```bash
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```

## 准备数据集

本模型输入prompt生成视频，无需数据集。

## 模型推理

1. 下载模型

   ST-DIT权重文件下载链接如下，按需下载：

   [ST-DIT-256x256下载链接](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth)

   [ST-DIT-512x512下载链接](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)

   vae权重文件下载链接如下，按需下载：

   ```bash
   # ema
   git clone https://huggingface.co/stabilityai/sd-vae-ft-ema
   ```

   encoder权重文件

   ```bash
   https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main
   ```

2. 模型转换，该步骤会生成编译之后的pt模型

   ```bash
   python3 export_model.py \
   --output_dir ./models \
   --encoder_model_path ./DeepFloyd--t5-v1_1-xxl \
   --dit_model_path ./OpenSora-v1-HQ-16x512x512.pth \
   --vae_model_path ./sd-vae-ft-ema \
   --resolution 16x512x512 \
   --device_id 0
   ```

   参数说明：

   - --encoder_model_path：encoder的权重路径
   - --dit_model_path：dit的权重路径
   - --vae_model_path：vae的权重路径
   - --resolution：分辨率。支持256和512
   - --device_id：NPU芯片
   - --output_dir：pt模型输出目录

3. 开始推理

   1. 开启cpu高性能模式

      ```bash
      echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
      sysctl -w vm.swappiness=0
      sysctl -w kernel.numa_balancing=0
      ```

   2.  执行推理，会在当前路径生成sample.mp4

      ```bash
      python inference.py \
      ./configs/opensora/inference/16x256x256.py \
      --ckpt-path ./OpenSora-v1-HQ-16x512x512.pth \
      --prompt-path ./assets/texts/t2v_samples.txt \
      --use_mindie 1 \
      --device_id 0
      ```
      
      参数说明：
      
      - --ckpt-path：STDIT的权重路径
      - --prompt-path：prompt数据集的路径
      - --use_mindie：是否使用MindIE推理。1代表是，0代表否
      - --device_id：使用哪张卡

# 模型推理性能

性能参考下列数据。

| 分辨率 | 硬件形态 | 平均耗时 |
| ------ | -------- | -------- |
| 512    | Atlas 800I A2 (32G) | 110.8s   |
| 256    | Atlas 300I Duo  | 22.2s    |
