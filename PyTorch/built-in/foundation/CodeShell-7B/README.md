# CodeShell-7B for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [CodeShell](#CodeShell)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

CodeShell是北京大学知识计算实验室联合四川天府银行AI团队研发的多语言代码大模型基座。CodeShell具有70亿参数，在五千亿Tokens进行了训练，上下文窗口长度为8194。在权威的代码评估Benchmark（HumanEval与MBPP）上，CodeShell取得同等规模最好的性能。与此同时，我们提供了与CodeShell配套的部署方案与IDE插件，请参考代码库CodeShell。

## 支持任务列表
本仓已经支持以下模型任务类型

|      模型      | 任务列表 | 是否支持 |
|:------------:|:----:| :------: |
| CodeShell-7B |  微调  |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/WisdomShell/codeshell
  commit_id=a33262e348eff888a28dd7226ee11ebc083c9df0
  ```
- 适配昇腾AI处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/foundation
    ```

# CodeShell

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

| 三方库  | 支持版本(PT2.1) | 支持版本(PT2.4) |
| :-----: |:-----------:|:-----------:|
| PyTorch |    2.1.0    |    2.4.0    |
| accelerate |   0.29.3    |   0.29.3    |
| deepspeed |   0.12.6    |   0.15.3    |
| transformers |   4.40.1    |   4.40.1    |

- 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

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
| FrameworkPTAdaper | 6.0.RC2  |
|       CANN        | 8.0.RC2  |
|    昇腾NPU固件    | 24.1.RC2 |
|    昇腾NPU驱动    | 24.1.RC2 |

### 准备预训练权重

- CodeShell-7B预训练权重需要在官网手动下载（<u>**除了modeling_codeshell.py**</u>），并放于model目录下。

  model参考目录如下：
   ```
   ├── model
         ├──added_tokens.json
         ├──config.json
         ├──configuration_codeshell.py
         ├──generation_config.json
         ├──merges.txt
         ├──model-00001-of-00002.safetensors
         ├──model-00002-of-00002.safetensors
         ├──model.safetensors.index.json
         ├──modeling_codeshell.py
         ├──pytorch_model.bin.index.json
         ├──quantizer.py
         ├──special_tokens_map.json
         ├──tokenizer.json
         ├──tokenizer_config.json
         ├──vocab.json
   ```


### 准备数据集

- 自行下载准备alpaca_data.json数据集，并放于finetune目录下。
- 在CodeShell-7B根目录下，执行以下python命令：
```
python convert_alpaca.py --in-file finetune/alpaca_data.json --out-file finetune/data.json
```
  处理后的数据集目录结构如下所示：
   ```
   ├── finetune
         ├──alpaca_data.json(原始数据集)
         ├──data.json(微调用数据集)
   ```


> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。      


## 快速开始
### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  1. 进入源码根目录。

     ```
     cd /${模型文件夹名称}
     ```

  2. 运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡训练

     ```
     bash finetune/run_finetune.sh --model=./model --data_path=./finetune/data.json --exp_id=0
     ```
     
     模型训练脚本参数说明如下。
     
     ```
     --model                 //预训练权重路径
     --data_path             //数据集路径
     --exp_id                //标识符，用于日志打印和模型保存的路径区分，可自定义
     ```


#### 训练结果
训练loss与train_samples_per_second可在训练日志中获取，其结果如下：

| 芯片       | 卡数       | Batch size | Steps | Train_Samples_Per_Second |
|----------|:--------:|:----------:|:-----:|:------------------------:|
| GPU      |    8p    |     6      | 2000  |          40.952          |
| Atlas-A2 |    8p    |     6      | 2000  |          36.801          |




# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

2024.05.16：首次发布。

# FAQ

无