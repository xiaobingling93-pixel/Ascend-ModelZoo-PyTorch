# OpenSora for Pytorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [STDiT2（在研版本）](#STDiT2（在研版本）)   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [训练任务](#训练任务)
          - [推理任务](#推理任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

OpenSora是HPC AI Tech开发的开源高效复现类Sora视频生成方案。OpenSora不仅实现了先进视频生成技术的低成本普及，还提供了一个精简且用户友好的方案，简化了视频制作的复杂性。
本仓库主要将STDiT2模型的任务迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|     模型      | 任务列表 | 是否支持 |
|:-----------:|:----:|:-----:|
| STDiT2-XL/2 | 在线训练 | ✔ |
| STDiT2-XL/2 | 在线推理 | ✔ |


## 代码实现

- 参考实现：

  ```
  url=https://github.com/hpcaitech/Open-Sora
  commit_id=74b645350b0f7a0ed802f87243c23edd1504c26d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/OpenSora
  ```


# STDiT2（在研版本）

## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  |     三方库     |  支持版本  |
  |:-----------:|:------:|
  |   PyTorch   | 2.1.0  |
  | TorchVision | 0.16.0 |


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```python
   source ${cann_install_path}/ascend-toolkit/set_env.sh              # 激活cann环境
   cd OpenSora
   pip install -v -e .                                                # 安装本地代码仓，同时自动安装依赖
   ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

  | 软件类型   |   支持版本   |
  | :--------: |:--------:|
  | FrameworkPTAdapter |   在研版本   |
  | CANN |   在研版本   |
  | 昇腾NPU固件 |   在研版本   | 
  | 昇腾NPU驱动 | 在研版本 |

  

### 准备数据集
#### 训练数据集准备
数据集准备请参考官网，链接如下：
https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#data-processing

### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   stabilityai/sd-vae-ft-ema   # vae模型
   DeepFloyd/t5-v1_1-xxl       # t5模型
   hpcai-tech/OpenSora-STDiT-v2-stage2        # 预训练权重(推理用)
   hpcai-tech/OpenSora-STDiT-v2-stage3        # 预训练权重(推理用)
   ```

   > **说明：**  
   > 在线推理时，对`hpcai-tech/OpenSora-STDiT-v2-stage3`和`hpcai-tech/OpenSora-STDiT-v2-stage3`模型需做一些离线转换，转换成.pth格式。提供参考用例：
   > ```python
   > import os
   > import torch
   > import safetensors
   > data = safetensors.torch.load_file('./hpcai-tech/OpenSora-STDiT-v2-stage2/model.safetensors')
   > data["state_dict"] = data
   > torch.save(data, os.path.splitext('./hpcai-tech/OpenSora-STDiT-v2-stage2/model.safetensors')[0]+'.pth')
   > ```   


3. 获取对应的预训练模型后，在以下配置文件中将`model`、`vae`的`from_pretrained`参数设置为本地预训练模型绝对路径。
   ```shell
   configs/opensora-v1-1/inference/sample.py
   configs/opensora-v1-1/train/stage1.py
   configs/opensora-v1-1/train/stage2.py
   configs/opensora-v1-1/train/stage3.py
   ```

4. 将下载好的t5模型放在本工程目录下的`DeepFloyd`目录下，组织结构如下：
   ```
   $OpenSora
   ├── DeepFloyd
   ├── ├── t5-v1_1-xxl
   ├── ├── ├── config.json
   ├── ├── ├── pytorch_model-00001-of-00002.bin
   ├── ├── ├── ...
   └── ...
   ```

## 快速开始
### 训练任务
本任务主要以预训练模型为主，展示训练任务，包含单机单卡和单机多卡的训练。
#### 开始训练
1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称} 
   ```
2. 准备训练数据。
按照官网流程，准备对应数据集，处理数据并得到包含数据信息的csv文件，放在模型文件夹下，如图：
   ```
   $OpenSora
   ├── train_data.csv
   └── ...
   ```

2. 运行训练脚本。

   用户可以按照自己训练需要进行参数配置，以下给出单卡和多卡的一种训练示例。
   ```shell
   bash test/infer_full_1p_opensorav1_1.sh --data_path=train_data.csv
   # 混合精度BF16，单卡训练，stage1
   ```

   ```shell
   bash test/infer_full_8p_opensorav1_1.sh --data_path=train_data.csv
   # 混合精度BF16，八卡训练，stage1
   ```

### 推理任务
本任务主要以预训练模型为主，展示推理任务，包括单卡在线推理。
#### 开始推理
1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称} 
   ```


2. 运行推理的脚本。

- 单机单卡推理
  ```shell
  bash test/infer_full_1p_opensorav1_1.sh --ckpt_path=/path/to/OpenSora-STDiT-v2-stage3/model.pth  # 混精bf16 在线推理
  ```
- 推理脚本参数说明如下
   ```shell
   test/infer_full_1p_opensorav1_1.sh
   --batch_size                         //设置batch_size
   --ckpt_path                          //推理加载的模型地址
   --prompt                             //测试用的prompt
   --num_frames                         //生成视频的总帧数
   --img_h                              //生成视频的宽
   --img_w                              //生成视频的高
  
   scripts/inference.py
   config                               //配置文件路径
   --seed                               //随机种子
   --ckpt-path                          //推理加载的模型文件路径    
   --batch-size                         //设置batch_size
   --prompt-path                        //推理使用的prompt文件路径
   --prompt                             //测试用的prompt
   --num-frames                         //生成视频的总帧数
   --image-size                         //生成视频的分辨率
   --fps                                //生成视频的帧率
   --save-dir                           //输出视频的路径
   --num-sampling-steps                 //推理的采样步数
   --cfg-scale                          //无分类器引导的权重系数
   ```



# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.04.29：OpenSora STDiT2 bf16训练和推理任务首次发布。

# FAQ



