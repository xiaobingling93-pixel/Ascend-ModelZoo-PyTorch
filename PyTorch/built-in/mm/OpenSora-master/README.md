
# OpenSora for PyTorch
# 目录

- [OpenSora for PyTorch](#opensora-for-pytorch)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备环境](#准备环境)
  - [安装模型环境](#安装模型环境)
  - [安装昇腾环境](#安装昇腾环境)
- [准备数据集](#准备数据集)
  - [训练数据集准备](#训练数据集准备)
  - [训练数据处理](#训练数据处理)
- [STDiT2（在研版本）](#stdit2在研版本)
  - [获取预训练模型](#获取预训练模型)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
        - [性能](#性能)
    - [推理任务](#推理任务)
      - [开始推理](#开始推理)
- [STDiT3 （在研版本）](#stdit3-在研版本)
  - [获取预训练模型](#获取预训练模型-1)
  - [快速开始](#快速开始-1)
    - [训练任务](#训练任务-1)
      - [开始训练](#开始训练-1)
    - [推理任务](#推理任务-1)
      - [开始推理](#开始推理-1)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
  - [变更](#变更)
- [FAQ](#faq)


# 简介
## 模型介绍

OpenSora是HPC AI Tech开发的开源高效复现类Sora视频生成方案。OpenSora不仅实现了先进视频生成技术的低成本普及，还提供了一个精简且用户友好的方案，简化了视频制作的复杂性。
本仓库主要将OpenSora1.1的STDiT2模型和OpenSora1.2的STDiT3模型的任务迁移到了昇腾NPU上，并进行极致性能优化。

> 注：OpenSora主线从此目录演进，包含OpenSora1.1和OpenSora1.2，OpenSora1.0单独维护，见 [PyTorch/built-in/mlm/OpenSora1.0 · Ascend/ModelZoo-PyTorch - 码云 - 开源中国 (gitee.com)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mlm/OpenSora1.0)


> <span style="color: red;">**注意**: OpenSora-master目录下面的OpenSora1.2模型已经集成到[MindSpeed-MM](https://gitee.com/ascend/MindSpeed-MM)中,当前目录下的OpenSora1.2模型不再维护，MindSpeed-MM是面向大规模分布式训练的昇腾多模态大模型套件，同时支持多模态生成及多模态理解,欢迎大家使用。</span>
## 支持任务列表

本仓已经支持以下模型任务类型

|     模型      | 任务列表 | 是否支持 |
|:-----------:|:----:|:-----:|
| STDiT2-XL/2 | 预训练 | ✔ |
| STDiT2-XL/2 | 在线推理 | ✔ |
| STDiT3-XL/2 | 预训练 | ✔ |
| STDiT3-XL/2 | 在线推理 | ✔ |


## 代码实现

- 参考实现：

  ```
  url=https://github.com/hpcaitech/Open-Sora
  commit_id=1295446e08ecdfe3a42ac93efd876dffacc76d5f
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mm/OpenSora-master
  ```

# 准备环境

##  安装模型环境


  **表 3**  三方库版本支持表

|   三方库    | 支持版本 |
| :---------: | :------: |
|   PyTorch   |  2.1.0   |
| TorchVision |  0.16.0  |


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```bash
source ${cann_install_path}/ascend-toolkit/set_env.sh              # 激活cann环境
cd OpenSora-master
pip install -v -e .                                                # 安装本地代码仓，同时自动安装依赖
   ```

   安装mindspeed：

   ```bash
git clone https://gitee.com/ascend/MindSpeed.git
pip install -e MindSpeed
   ```

## 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表4中软件版本。                

  **表 4**  昇腾软件版本支持表

|      软件类型      | 支持版本 |
| :----------------: | :------: |
| FrameworkPTAdapter | 在研版本 |
|        CANN        | 在研版本 |
|    昇腾NPU固件     | 在研版本 |
|    昇腾NPU驱动     | 在研版本 |

# 准备数据集

## 训练数据集准备

训练数据集准备请参考官网，链接如下：

https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#data-processing

## 训练数据处理
处理数据并得到包含数据信息的csv文件，放在模型文件夹下，如下：
  ```
  $OpenSora-master
  ├── train_data.csv
  └── ...
  ```
其中train_data.csv中必须要包含的字段及含义如下：
  ```bash
  path		# 视频绝对路径
  text		# 视频文本标注
  num_frames	# 视频帧数
  width			# 视频宽度
  height		# 视频高度
  ```

# STDiT2（在研版本）

## 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   PixArt-alpha/PixArt-alpha   # PixArt-XL-2-512x512模型(训练用)
   stabilityai/sd-vae-ft-ema   # vae模型
   DeepFloyd/t5-v1_1-xxl       # t5模型
   hpcai-tech/OpenSora-STDiT-v2-stage2        # 预训练权重(推理用)
   hpcai-tech/OpenSora-STDiT-v2-stage3        # 预训练权重(推理用)
   ```

3. 获取对应的预训练模型后，在以下配置文件中将`model`、`vae`的`from_pretrained`参数设置为本地预训练模型绝对路径。
   ```shell
   configs/opensora-v1-1/inference/sample.py
   configs/opensora-v1-1/train/stage1.py
   configs/opensora-v1-1/train/stage2.py
   configs/opensora-v1-1/train/stage3.py
   ```

4. 将下载好的t5模型放在本工程目录下的`DeepFloyd`目录下，组织结构如下：
   ```
   $OpenSora-master
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
    按照官网流程，准备对应数据集，获取对应数据集后，在以下配置文件中将`dataset`的`data_path`设置为数据集的绝对路径
    ```bash
    configs/opensora-v1-1/train/stage1.py
    configs/opensora-v1-1/train/stage2.py
    configs/opensora-v1-1/train/stage3.py
    ```

3. 运行训练脚本。

   用户可以按照自己训练需要进行参数配置，以下给出单卡和多卡的一种训练示例。
   
   ```shell
   bash test/train_full_1p_opensorav1_1.sh
   # 混合精度BF16，单卡训练，stage1
   ```
   
   ```shell
   bash test/train_full_8p_opensorav1_1.sh
   # 混合精度BF16，八卡训练，stage1
   ```
#### 训练结果
##### 性能

| Stage  | 芯片          | 卡数 | 平均单步耗时 | AMP_Type | Torch_Version |
| ------ | ------------- | ---- | ------------ | -------- | ------------- |
| Stage1 | 竞品          | 8    | 1.59 s       | bf16     | 2.1           |
| Stage1 | Atlas 800T A2 | 8    | 1.46 s       | bf16     | 2.1           |
| Stage2 | 竞品          | 8    | 11.10 s      | bf16     | 2.1           |
| Stage2 | Atlas 800T A2 | 8    | 12.97 s      | bf16     | 2.1           |
| Stage3 | 竞品          | 8    | 16.12 s      | bf16     | 2.1           |
| Stage3 | Atlas 800T A2 | 8    | 18.97 s      | bf16     | 2.1           |

> 注：动态分辨率端到端耗时与数据集强相关，该性能仅供参考，stage1/2/3对应训练的三个阶段，可以在train_full_8p_opensorav1_1.sh中修改

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
  bash test/infer_full_1p_opensorav1_1.sh  # 混精bf16 在线推理
  ```
- 推理脚本参数说明如下
   ```shell
   test/infer_full_1p_opensorav1_1.sh
   --batch_size                         //设置batch_size
   --prompt                             //测试用的prompt
   --num_frames                         //生成视频的总帧数
   --img_h                              //生成视频的宽
   --img_w                              //生成视频的高
     
  scripts/inference.py
   config                               //配置文件路径
   --seed                               //随机种子    
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

# STDiT3 （在研版本）

## 获取预训练模型
1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   PixArt-alpha/PixArt-Sigma   		# PixArt-Sigma-XL-2-2K-MS模型(训练用)
   PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers	# vae模型
   hpcai-tech/OpenSora-VAE-v1.2   		# vae模型
   DeepFloyd/t5-v1_1-xxl      	 		# t5模型
   hpcai-tech/OpenSora-STDiT-v3        # 预训练权重(推理用)
   ```

3. 获取对应的预训练模型后，在以下配置文件中将`model`、`vae`的`from_pretrained`参数设置为本地预训练模型绝对路径。
   ```shell
   configs/opensora-v1-2/inference/sample.py
   configs/opensora-v1-2/train/stage1.py
   configs/opensora-v1-2/train/stage2.py
   configs/opensora-v1-2/train/stage3.py
   ```

4. 将下载好的`t5`模型放在本工程目录下的`DeepFloyd`目录下，组织结构如下：
   ```
   $OpenSora-master
   ├── DeepFloyd
   ├── ├── t5-v1_1-xxl
   ├── ├── ├── config.json
   ├── ├── ├── pytorch_model-00001-of-00002.bin
   ├── ├── ├── ...
   └── ...
   ```
   
   将下载好的`pixart_sigma_sdxlvae_T5_diffusers`模型放在`PixArt-alpha`目录下（只需要有其中的`vae`文件夹）
   
   ```
   $OpenSora-master
   ├── PixArt-alpha
   ├── ├── pixart_sigma_sdxlvae_T5_diffusers
   ├── ├── ├── vae
   ├── ├── ├── ├── config.json
   ├── ├── ├── ├── ...
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
    按照官网流程，准备对应数据集，获取对应数据集后，在以下配置文件中将`dataset`的`data_path`设置为数据集的绝对路径
    ```bash
    configs/opensora-v1-2/train/stage1.py
    configs/opensora-v1-2/train/stage2.py
    configs/opensora-v1-2/train/stage3.py
    ```

3. 运行训练脚本。

   用户可以按照自己训练需要进行参数配置，以下给出单卡和多卡的一种训练示例。
   ```shell
   bash test/train_full_1p_opensorav1_2.sh
   # 混合精度BF16，单卡训练，stage1
   ```

   ```shell
   bash test/train_full_8p_opensorav1_2.sh
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
  bash test/infer_full_1p_opensorav1_2.sh  # 混精bf16 在线推理
  ```
- 推理脚本参数说明如下
   ```shell
   test/infer_full_1p_opensorav1_2.sh
   --batch_size                         //设置batch_size
   --prompt                             //测试用的prompt
   --num_frames                         //生成视频的总帧数
   --img_h                              //生成视频的宽
   --img_w                              //生成视频的高
     
  scripts/inference.py
   config                               //配置文件路径
   --seed                               //随机种子    
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

2024.04.29：OpenSora1.1 STDiT2 bf16训练和推理任务首次发布。

2024.08.13：OpenSora1.2 STDiT3 bf16训练和推理任务首次发布。

# FAQ



