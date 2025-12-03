
# HunyuanDiT for PyTorch

# 目录

- [HunyuanDiT for PyTorch](#hunyuandit-for-pytorch)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [HunyuanDiT（在研版本）](#HunyuanDiT在研版本)
  - [准备训练环境](#准备训练环境)
    - [安装模型环境](#安装模型环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
      - [训练数据集准备](#训练数据集准备)
    - [获取预训练模型](#获取预训练模型)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
    - [推理任务](#推理任务)
      - [开始推理](#开始推理)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
  - [变更](#变更)
- [FAQ](#faq)

# 简介

## 模型介绍

HunyuanDiT是由腾讯开发并开源的一款先进的文生图（文本到图像）模型。该模型支持中英文双语输入，特别针对中文进行了优化，能够深刻理解中文语境和文化元素，生成高质量且富有中国文化特色的图像。HunyuanDiT经过大规模中文数据集的训练，涵盖了广泛的类别和艺术风格，能够根据文本提示生成细腻逼真的图像。
本仓库主要将HunyuanDiT模型的任务迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|     模型      | 任务列表 | 是否支持 |
|:-----------:|:----:|:-----:|
| DiT-g/2 | 在线训练 | ✔ |
| DiT-g/2 | 在线推理 | ✔ |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/Tencent/HunyuanDiT
  commit_id=3bb80e1dedba5bf9728e7c9566c4b5c665bbfbd2
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/HunyuanDiT
  ```

# HunyuanDiT（在研版本）

## 准备训练环境

### 安装模型环境

  **表 3**  三方库版本支持表

  |     三方库     |  支持版本  |
  |:-----------:|:------:|
  |   PyTorch   | 2.1.0  |
  | TorchVision | 0.16.0 |
  | deepspeed | 0.14.4 |
  | diffusers | 0.21.2 |
| transformers | 4.39.1 |
| accelerate | 0.27.2 |

   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

   ```python
   source ${cann_install_path}/ascend-toolkit/set_env.sh              # 激活cann环境
   cd HunyuanDiT
   pip install -r requirements.txt                                    #安装其它依赖
   ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表4中软件版本。

  **表 4**  昇腾软件版本支持表

  | 软件类型   |   支持版本   |
  | :--------: |:--------:|
  | FrameworkPTAdapter |   在研版本   |
  | CANN |   在研版本   |
  | 昇腾NPU固件 |   在研版本   |
  | 昇腾NPU驱动 | 在研版本 |

### 准备数据集

#### 训练数据集准备

数据集准备请参考官网，链接如下：
<https://github.com/Tencent/HunyuanDiT>

### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载(<https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main>)

3. 将下载好的t5模型放在本工程目录下的`ckpts`目录下，组织结构如下：

   ```
   $HunyuanDiT
   ├── ckpts
   ├── ├── t2i
   ├── ├── ├── clip_text_encoder
   ├── ├── ├── model
   ├── ├── ├── mt5
   ├── ├── ├── sdxl-vae-fp16-fix
   ├── ├── ├── tokenizer
   └── ...
   ```

## 快速开始

### 训练任务

本任务主要以全参微调为主，展示训练任务，包含单机单卡和单机多卡的训练。

#### 开始训练

1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称} 
   ```

2. 准备训练数据。
按照官网流程，准备对应数据集，放在模型文件夹下，如图：

   ```
   dataset
    ├──porcelain
    │  ├──images/  (image files)
    │  │  ├──0.png
    │  │  ├──1.png
    │  │  ├──......
    │  ├──csvfile/  (csv files containing text-image pairs)
    │  │  ├──image_text.csv
    │  ├──arrows/  (arrow files containing all necessary training data)
    │  │  ├──00000.arrow
    │  │  ├──00001.arrow
    │  │  ├──......
    │  ├──jsons/  (final training data index files which read data from arrow files during training)
    │  │  ├──porcelain.json
    │  │  ├──porcelain_mt.json
   ```

2. 运行训练脚本。

   用户可以按照自己训练需要进行参数配置，以下给出多卡的一种训练示例。

   【如需长步数训练】
   需修改epochs默认步数1400步为所需步数 (Total Optimzation Steps为Epochs * len(Data Loader) // Gradient Accumulation Steps)

   ```shell
   vim test/train_full_8p_bf16.sh

   # 修改epochs为所需步数
   --reso-step 64 \
   --epochs 1400 \ 
   --max-training-steps ${max_train_steps} \
   ```
   
   ```shell
   vim test/train_full_8p_bf16.sh

   # 修改epochs为所需步数
   --reso-step 64 \
   --epochs 1400 \ 
   --max-training-steps ${max_train_steps} \
   ```
   
   【zero3】配置可在`/hydit/modules/models.py`文件里修改层数等，如修改depth层为90

   运行zero2配置脚本

   ```shell
   bash test/train_full_8p_bf16.sh
   # 混合精度BF16，8卡训练
   ```
   
   运行zero3配置脚本
   ```shell
   bash test/train_full_8p_bf16_zero3.sh
   # 混合精度BF16，8卡训练
   ```

### 性能展示

#### 性能 (zero2)

 |    芯片    | 卡数 | 单步迭代耗时（ms/step） | batch_size | AMP_dtype |
  |:--------:|:--:|-----------------|------------|-----------|
  |   GPU    | 8p |        1059.3   |     1      |    BF16   |
  | Atlas A2 | 8p |         1011.7  |      1     |    BF16   |

### 推理任务

本任务主要以全参微调为主，展示推理任务，包括单卡在线推理。

#### 开始推理

1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称} 
   ```

2. 运行推理的脚本（待补充）。

- 单机单卡推理

  ```shell
  bash test/inference_full_1p_fp16.sh  # 混精fp16 在线推理
  ```

- 推理脚本参数说明如下

   ```shell
   test/inference_full_1p_fp16.sh
   --prompt                         //测试用的prompt
   ```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.08.22：HunyuanDiT bf16训练和fp16推理任务首次发布。

# FAQ
