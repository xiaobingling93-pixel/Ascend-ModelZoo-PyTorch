# DiT模型-推理指导

# 概述

DiT一种基于Transformer的扩散模型，全称为Diffusion Transformer，DiT遵循ViT的技术方法。有关DiT模型的更多信息，请参考[DiT github](https://github.com/facebookresearch/DiT)。

- 设备支持：
Atlas 800I A2推理设备
Atlas 300I Duo推理卡

## 输入输出数据

image_num为需要生成的图片数量

batch_size = image_num * 2

latent_size = image_size // 8

- 输入数据

  | 输入数据 | 数据类型 | 大小                                       | 数据排布格式 |
  | -------- | -------- | ------------------------------------------ | ------------ |
  | x        | FLOAT32  | batch_size x 4 x latent_size x latent_size | NCHW         |
  | t        | INT64    | batch_size                                 | ND           |
  | y        | INT64    | batch_size                                 | ND           |


- 输出数据

  | 输出数据 | 数据类型 | 大小                                      | 数据排布格式 |
  | -------- | -------- | ----------------------------------------- | ------------ |
  | output   | FLOAT32  | image_num x 4 x latent_size x latent_size | NCHW         |

# 推理环境准备

**表 1**  版本配套表

| 配套                                 | 版本    | 环境准备指导 |
| ------------------------------------ | ------- | ------------ |
| Python                               | 3.10.13 | -            |
| PyTorch                              | 2.1.0   | -            |
| 硬件：Atlas 300I Duo, Atlas 800I A2 | \       | \            |

请以CANN版本选择对应的固件与驱动版本。

# 快速上手

## 获取源码

1. 获取源码，然后把当前目录下的几个文件移到DiT工程下

   ```bash
   git clone https://github.com/facebookresearch/DiT
   mv background_runtime.py export_model.py models_npu.py sample_npu.py vision.patch timm_patch.py requirements.txt fid_test.py ./DiT
   ```

2. 安装依赖

   ```bash
   pip3 install -r requirements.txt

   ```

3. 安装mindie包

   ```bash
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```

4. 代码修改

   ```
   cd ./DiT
   # 若环境没有patch工具，请自行安装
   python3 timm_patch.py
   ```

## 准备数据集

本模型输入图片类别信息生成图片，无需数据集。

## 模型推理

1. 下载模型

   DiT权重文件下载链接如下，按需下载：

   [DiT-XL-2-256x256下载链接](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)

   [DiT-XL-2-512x512下载链接](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt)

   vae权重文件下载链接如下，按需下载：

   ```bash
   # ema
   git clone https://huggingface.co/stabilityai/sd-vae-ft-ema
   # mse
   git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
   ```

2. 模型转换，该步骤会生成编译之后的pt模型

   ```bash
   # Atlas 300I Duo卡
   python3 export_model.py \
   --ckpt ./DiT-XL-2-512x512.pt \
   --vae_model ./sd-vae-ft-mse \
   --image_size 512 \
   --device 0 \
   --soc Duo \
   --output_dir ./models \
   --parallel
   
   # Atlas 800I A2
   python3 export_model.py \
   --ckpt ./DiT-XL-2-512x512.pt \
   --vae_model ./sd-vae-ft-mse \
   --image_size 512 \
   --device 0 \
   --soc A2 \
   --output_dir ./models
   ```

   参数说明：

   - --ckpt：DiT-XL-2的权重路径
   - --vae_model：vae的权重路径
   - --image_size：分辨率，支持256和512。默认为512
   - --device：使用哪张卡
   - --soc：soc_version，只支持Duo和A2
   - --output_dir：pt模型输出目录
   - --parallel：【可选】模型使用并行进行推理

3. 开始推理

   1. 开启cpu高性能模式

      ```bash
      echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
      sysctl -w vm.swappiness=0
      sysctl -w kernel.numa_balancing=0
      ```

   2.  执行推理，会在当前路径生成sample.png

      ```bash
      # Atlas 300I Duo
      python3 sample_npu.py \
      --vae mse \
      --image_size 512 \
      --ckpt ./DiT-XL-2-512x512.pt \
      --device 0 \
      --class_label 0 \
      --output_dir ./models \
      --parallel
      
      # Atlas 800I A2
      python3 sample_npu.py \
      --vae mse \
      --image_size 512 \
      --ckpt ./DiT-XL-2-512x512.pt \
      --device 0 \
      --class_label 0 \
      --output_dir ./models \
      --warmup
      ```

      参数说明：

      - --vae：使用哪种vae模型，支持mse和ema
      - --image_size：分辨率，支持256和512。默认为512
      - --ckpt：DiT-XL-2的权重路径
      - --device：使用哪张卡
      - --class_label：可在0~999中任意指定一个整数，代表image_net的种类
      - --output_dir：上一步骤指定的pt模型输出目录
      - --parallel：【可选】模型使用并行进行推理
      - --warmup:【可选】使用warmup可使得时间更准确。并行场景使用该选项会有问题，不建议使用

4. 精度验证

   下载数据集[ImageNet512x512](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz)(VIRTUAL_imagenet512.npz)和[ImageNet256x256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)(VIRTUAL_imagenet256_labeled.npz)，放在任意路径

   然后执行以下命令：

   ```bash
   # Atlas 300I Duo
   python3 fid_test.py \
   --vae mse \
   --image_size 512 \
   --ckpt ./DiT-XL-2-512x512.pt \
   --device 0 \
   --output_dir ./models \
   --parallel \
   --results results
   
   # Atlas 800I A2
   python3 fid_test.py \
   --vae mse \
   --image_size 512 \
   --ckpt ./DiT-XL-2-512x512.pt \
   --device 0 \
   --output_dir ./models \
   --results results
   ```

   参数说明：

   - --results：生成的1000张图片存放路径
   - image_size：分辨率，支持256和512。默认为512
   
   之后进行FID计算：
   
   ```bash
   # 512分辨率使用VIRTUAL_imagenet512.npz数据集
   python3 -m pytorch_fid ./VIRTUAL_imagenet512.npz ./results
   # 256分辨率使用VIRTUAL_imagenet256_labeled.npz数据集
   python3 -m pytorch_fid ./VIRTUAL_imagenet256_labeled.npz ./results 
   ```

# 模型推理性能&精度

性能参考下列数据。

| 分辨率 | 硬件形态 | 迭代次数 | 平均耗时 |
| ------ | -------- | -------- | -------- |
| 512    | Atlas 300I Duo      | 250      | 19.6s    |
|        | Atlas 800I A2 (32G) | 250      | 10.49s   |
| 256    | Atlas 300I Duo      | 250      | 9.5s     |
|        | Atlas 800I A2 (32G) | 50       | 4.13s    |
