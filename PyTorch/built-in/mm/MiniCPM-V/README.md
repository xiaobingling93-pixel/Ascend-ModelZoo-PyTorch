
# MiniCPM-V for PyTorch



# 目录
- [MiniCPM-V](#MiniCPM-V-for-pytorch)
  - [概述](#概述)
  - [准备训练环境](#准备训练环境)
    - [创建Python环境](#创建python环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [模型训练](#模型训练)
    - [结果展示](#结果展示)
    - [模型推理](#模型推理)
  - [公网地址说明](#公网地址说明)
  - [变更说明](#变更说明)
  - [FQA](#faq)



## 概述

### 模型介绍

MiniCPM-V是面向图文理解的端侧多模态大模型系列。该系列模型接受图像和文本输入，并提供高质量的文本输出。MiniCPM-Llama3-V 2.5的多模态综合性能超越 GPT-4V-1106、Gemini Pro、Claude 3、Qwen-VL-Max 等商用闭源模型，OCR 能力及指令跟随能力进一步提升，并支持超过30种语言的多模态交互。
### 支持任务列表
本仓已经支持以下模型任务类型

|      模型      |  任务列表  | 是否支持 |
|:------------:|:------:|:-----:|
|  MiniCPM-V   |  全参微调  | ✔ |
| MiniCPM-V | Lora微调 | ✔ |
| MiniCPM-V |  在线推理  | ✔ |

### 代码实现
- 参考实现：

  ```
  url=https://github.com/OpenBMB/MiniCPM-V.git
  commit_id=6a5f9a4d6556e47767e7b653a9279281d2ef7062
  ```

- 适配昇腾AI处理器的实现：
  ```shell
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/MiniCPM-V
  ```

## 准备训练环境

### 创建Python环境

- git clone 远程仓
  ```shell
  git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
  cd PyTorch/built-in/mlm/MiniCPM-V
  ```

- 创建Python环境并且安装Python三方包
  ```shell
  conda create -n MiniCPM-V python=3.10 -y
  conda activate MiniCPM-V
  pip install -r requirements.txt
  ```

- 环境准备指导

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

    **表 1**  昇腾软件版本支持表

  |     软件类型     |   支持版本   |
  |:-----------:|:--------:|
  | FrameworkPTAdapter  |   在研版本   |
  | CANN | 在研版本 |
   | 昇腾NPU固件 | 在研版本 |
   | 昇腾NPU驱动 | 在研版本 |


### 准备数据集

- 需要自行下载textVQA数据集，涉及到的数据集结构如下所示：
   ```
    TextVQA
      ├── TextVQA_0.5.1_train.json
      ├── TextVQA_0.5.1_val.json
      └── train_images
   ```
  json文件格式请参考 https://github.com/OpenBMB/MiniCPM-V/blob/main/finetune/readme.md 中的数据准备章节。
### 准备预训练权重

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：
    ```shell
    openbmb/MiniCPM-Llama3-V-2_5
    ```
3. 预训练模型文件下载后，需替换部分文件，使用下面命令时注意修改安装环境的路径。
    ```shell
    cp huggingface_modify/configuration_minicpm.py openbmb/MiniCPM-Llama3-V-2_5/configuration_minicpm.py
    cp huggingface_modify/modeling_minicpmv.py openbmb/MiniCPM-Llama3-V-2_5/modeling_minicpmv.py
    cp huggingface_modify/resampler.py openbmb/MiniCPM-Llama3-V-2_5/resampler.py
    ```
## 快速开始

### 模型训练

1. 全参微调脚本位置位于finetune/finetune_ds.sh；Lora微调脚本位置位于finetune/finetune_lora.sh，需要手动将数据集，权重的路径传入到相应参数上，路径仅供参考，请用户根据实际情况修改。
   ```shell
    MODEL="openbmb/MiniCPM-Llama3-V-2_5"  # MiniCPM-V权重路径
    DATA="path/to/trainging_data"  # 训练数据路径
    EVAL_DATA="path/to/test_data"  # 验证数据路径
   ```

2. 运行训练脚本，该模型支持单机8卡训练。

    ```shell
    bash finetune/finetune_ds.sh # 全参微调
    bash finetune/finetune_lora.sh # Lora微调
    ```
   训练完成后，权重文件保存在参数`--finetune/output`路径下。
### 结果展示

**表 2**  训练结果展示

|         芯片         | 卡数 | 50-200步训练耗时(s) | batch_size | Data_Type | Torch_Version |
|:------------------:|:---:|:--------------:|:----------:|:---------:|:---:|
|      竞品A-全参微调      | 8p |      847       |     12     |   bf16    | 2.1 |
| Atlas 800T A2-全参微调 | 8p |      1046      |     12     |   bf16    | 2.1 |
|     竞品A-Lora微调     | 8p |      490       |     8      |   bf16    | 2.1 |
|      Atlas 800T A2-Lora微调      | 8p |      603       |     8      |   bf16    | 2.1 |

### 模型推理

 执行下面命令即可进行推理。
   ```
  python web_demo_2.5.py --device npu
   ```

## 公网地址说明

代码涉及公网地址参考 public_address_statement.md


## 变更说明
2024.08.26: 首次发布

2024.08.29: 添加NPU适配代码

## FAQ
无
