# Qwen1.0 for PyTorch

## 目录

- [Qwen1.0 for PyTorch](#qwen10-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [Qwen1.0](#qwen10)
  - [准备训练环境](#准备训练环境)
    - [安装环境](#安装环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备预训练权重](#准备预训练权重)
    - [准备数据集](#准备数据集)
  - [快速开始](#快速开始)
    - [微调任务](#微调任务)
      - [开始微调](#开始微调)
      - [训练结果](#训练结果)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

Qwen是一个全能的语言模型系列，包含各种参数量的模型，如Qwen（基础预训练语言模型，即基座模型）和Qwen-Chat（聊天模型，该模型采用人类对齐技术进行微调）。基座模型在众多下游任务中始终表现出卓越的性能，而聊天模型，尤其是使用人类反馈强化学习（RLHF）训练的模型，具有很强的竞争力。聊天模型Qwen-Chat拥有先进的工具使用和规划能力，可用于创建agent应用程序。即使在使用代码解释器等复杂任务上，Qwen-Chat与更大的模型相比也能表现出极具竞争力的性能。

## 支持任务列表
本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| Qwen-1.8B-Chat | 微调 | ✔ |
| Qwen-7B-Chat   | 微调 | ✔ |
| Qwen-14B-Chat  | 微调 | ✔ |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/QwenLM/Qwen
  commit_id=a6c1ea82bad57338e07744ee983b5d9e7ca82426
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/foundation
    ```

# Qwen1.0

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

| 三方库  | 支持版本(PT2.1) | 支持版本(PT2.4) |
| :-----: |:-----------:|:-----------:|
| PyTorch |    2.1.0    |    2.4.0    |
| accelerate |   0.27.0    |   0.27.0    |
| deepspeed |   0.12.6    |   0.15.3    |
| transformers |   4.37.2    |   4.37.2    |

在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

```shell
# PyTorch 2.1请使用requirements_2_1.txt
pip install -r requirements_2_1.txt

# PyTorch 2.4请使用requirements_2_4.txt
pip install -r requirements_2_4.txt
```
> **说明：** 
>只需执行一条对应的PyTorch版本依赖安装命令。

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

  **表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.RC1  |
|       CANN        | 8.0.RC1  |
|    昇腾NPU固件    | 24.1.RC1 |
|    昇腾NPU驱动    | 24.1.RC1 |

### 准备预训练权重

请手动下载以下三个预训练权重保存至磁盘空间充足的地方：

- Qwen-1.8B-Chat
- Qwen-7B-Chat
- Qwen-14B-Chat

下载权重文件后文件夹结构如下所示：

```text
Qwen-1_8B-Chat
  ├── assets
  ├── examples
  ├── LICENSE
  ├── NOTICE
  ├── README.md
  ├── cache_autogptq_cuda_256.cpp
  ├── cache_autogptq_cuda_kernel_256.cu
  ├── config.json
  ├── configuration_qwen.py
  ├── cpp_kernels.py
  ├── generation_config.json
  ├── model-00001-of-00002.safetensors
  ├── model-00002-of-00002.safetensors
  ├── model.safetensors.index.json
  ├── modeling_qwen.py
  ├── qwen.tiktoken
  ├── qwen_generation_utils.py
  ├── tokenization_qwen.py
  ├── tokenizer_config.json

Qwen-7B-Chat
  ├── assets
  ├── examples
  ├── LICENSE
  ├── NOTICE
  ├── README.md
  ├── cache_autogptq_cuda_256.cpp
  ├── cache_autogptq_cuda_kernel_256.cu
  ├── config.json
  ├── configuration_qwen.py
  ├── cpp_kernels.py
  ├── generation_config.json
  ├── model-00001-of-00008.safetensors
  ├── model-00002-of-00008.safetensors
  ├── model-00003-of-00008.safetensors
  ├── model-00004-of-00008.safetensors
  ├── model-00005-of-00008.safetensors
  ├── model-00006-of-00008.safetensors
  ├── model-00007-of-00008.safetensors
  ├── model-00008-of-00008.safetensors
  ├── model.safetensors.index.json
  ├── modeling_qwen.py
  ├── qwen.tiktoken
  ├── qwen_generation_utils.py
  ├── tokenization_qwen.py
  ├── tokenizer_config.json

Qwen-14B-Chat
  ├── assets
  ├── examples
  ├── LICENSE
  ├── NOTICE
  ├── README.md
  ├── cache_autogptq_cuda_256.cpp
  ├── cache_autogptq_cuda_kernel_256.cu
  ├── config.json
  ├── configuration_qwen.py
  ├── cpp_kernels.py
  ├── generation_config.json
  ├── model-00001-of-00015.safetensors
  ├── model-00002-of-00015.safetensors
  ├── model-00003-of-00015.safetensors
  ├── model-00004-of-00015.safetensors
  ├── model-00005-of-00015.safetensors
  ├── model-00006-of-00015.safetensors
  ├── model-00007-of-00015.safetensors
  ├── model-00008-of-00015.safetensors
  ├── model-00009-of-00015.safetensors
  ├── model-00010-of-00015.safetensors
  ├── model-00011-of-00015.safetensors
  ├── model-00012-of-00015.safetensors
  ├── model-00013-of-00015.safetensors
  ├── model-00014-of-00015.safetensors
  ├── model-00015-of-00015.safetensors
  ├── model.safetensors.index.json
  ├── modeling_qwen.py
  ├── qwen.tiktoken
  ├── qwen_generation_utils.py
  ├── tokenization_qwen.py
  ├── tokenizer_config.json
```

### 准备数据集

自行下载准备alpaca_data数据集并放于模型根目录下。

在源码根目录下执行`convert_alpaca.py`，将原始数据集转换为指定格式并保存在当前目录下。

``` bash
python convert_alpaca.py --in alpaca_data.json --out alpaca_data_qwen.json
```

转换后格式样例：

```text
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "Give three tips for staying healthy."
      },
      {
        "from": "assistant",
        "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ]
  },
```

新生成的文件名为alpaca_data_qwen.json。目录结构如下所示：

```text
Qwen1.0
  ├── tests
  ├── alpaca_data_qwen.json
  ├── convert_alpaca.py
  ...    
```

> **说明：**  
> 该数据集的微调过程脚本只作为一种参考示例。      

## 快速开始
### 微调任务

本任务主要提供**单机8卡**和**双机16卡**的微调脚本。

#### 开始微调

  1. 进入源码根目录下tests文件夹。

     ```shell
     cd /${模型文件夹名称}/tests
     ```

  2. 运行微调脚本。

     该模型支持单机8卡和双机16卡微调。运行脚本前，请将代码中的MODEL替换成模型权重的绝对路径。

     - 单机8卡微调

     ```shell
     # Qwen-1.8B-Chat模型
     bash finetune_1_8B_Chat.sh

     # Qwen-7B-Chat模型
     bash finetune_7B_Chat.sh
     ```

     - 双机16卡微调
       
       请将config.yaml文件中main_process_ip改为主节点的ip，并将副节点中的machine_rank设置为1，两台机器上的代码、模型存放路径等完全相同。在两台机器上先后执行以下脚本。

      ```shell
      # Qwen-14B-Chat模型
      bash finetune_14B_Chat.sh 
      ```

#### 训练结果

  **表 3**  训练结果展示表

| 芯片 | 模型 | 卡数 | Batch size | Steps | Train_Samples_Per_Second |
|--------|:--------:|:--------:|:--------:|:--------:|:--------:|
| 竞品A      | 1.8B-Chat | 8p  | 8 | 2000  |          139.644         |
| Atlas 800T A2 | 1.8B-Chat | 8p  | 8 | 2000  |          130.413         |
| 竞品A      | 7B-Chat   | 8p  | 4 | 2000  |          33.631          |
| Atlas 800T A2 | 7B-Chat   | 8p  | 4 | 2000  |          28.675          |
| 竞品A      | 14B-Chat  | 16p | 1 | 2000  |          13.534          |
| Atlas 800T A2 | 14B-Chat  | 16p | 1 | 2000  |          12.838          |

# 公网地址说明

代码涉及公网地址参考public_address_statement.md

# 变更说明

2024.05.21：首次发布。

# FAQ

目前精度对比关闭了ds_config_zero2.json中的overlap_comm，开启overlap_comm功能会触发DeepSeed的原生问题，社区已有两个相关issue。我们提交了一个Pull Request，以供进一步审查和修复。以下是相关的链接：

- https://github.com/microsoft/DeepSpeed/issues/5523
- https://github.com/microsoft/DeepSpeed/issues/5545
- https://github.com/microsoft/DeepSpeed/pull/5606
  