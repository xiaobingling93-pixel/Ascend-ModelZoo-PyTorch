# 当前模型脚本已不随版本演进，如使用此模型可跳转至该[地址](https://gitee.com/ascend/MindSpeed-LLM)

# Qwen-7B for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [Qwen-7B](#Qwen-7B)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

Qwen-7B 是一个基于Transformer的大语言模型，在超大规模的预训练数据上进行训练得到，具有 70 亿参数。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

## 支持任务列表
本仓已经支持以下模型任务类型

|      模型      | 任务列表 | 是否支持 |
| :------------: | :------: | :------: |
|    Qwen-7B     |  预训练  |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/hiyouga/LLaMA-Factory/
  commit_id=27b04bce90b34e719375576cc67ff5374bb2f38a
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```

# Qwen-7B

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

|   三方库    | 支持版本 |
| :---------: | :------: |
|   PyTorch   |  2.1.0   |
| transformers|  4.34.1 |
|  deepspeed  |  0.14.0  |
  
- 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。



  ```shell
  pip install -r requirements.txt
  ```
  


- 修改三方库文件training_args.py


通过pip show pip命令查看三方库安装路径path，然后修改文件
vim path/transformers/training_args.py
注释1369-1394行

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境


### 准备预训练权重

- 用户手动下载模型权重文件（model-0000*-of-00008.safetensors），预训练依赖该模型权重，并将Qwen-7B/model/modeling_qwen.py文件替换已下载的权重模型文件modeling_qwen.py。
权重文件目录如下

  ```
  ├── model_weight
      ├──config.json
      ├──configuration_qwen.py
      ├──configuration.json
      ├──model-00001-of-00008.safetensors
      ├──model-00002-of-00008.safetensors
      ├──model-00003-of-00008.safetensors
      ├──model-00004-of-00008.safetensors
      ├──model-00005-of-00008.safetensors
      ├──model-00006-of-00008.safetensors
      ├──model-00007-of-00008.safetensors
      ├──model-00008-of-00008.safetensors
      ├──modeling_qwen.py
      ├──qwen.tiktoken
      ├──tokenization_qwen.py	
      ├──qwen_generation_utils.py
      ├──tokenizer_config.json
  ```




### 准备数据集

- 启动模型联网环境将自动下载数据集，手动获取数据集见模型开源代码目录LLaMA-Factory/data/wiki_demo.txt。

  ```
  ├──data
    ├──wiki_demo.txt
  ```

> **说明：**  
>  该数据集的训练过程脚本只作为一种参考示例。

## 快速开始

#### 开始训练

  1. 进入源码根目录。

     ```
     cd /${模型文件夹名称}
     ```

  2. 配置启动脚本。修改启动脚本qwen_7B_8p.sh或qwen_7B_64p.sh

     ```
     model_name="./model_weight"  //设置模型权重路径
     ```

  3. 运行训练脚本。

     - 单机8卡训练

     ```
     bash test/qwen_7B_8p.sh
     ```

     - 8机64卡训练

     ```
     bash test/qwen_7B_64p.sh
     ```

     模型训练脚本参数说明如下。

     ```
       --per_device_train_batch_size      //设置batch_size
       --bf16 \                           //设置数据类型
       --cutoff                           //设置seq_length
       --model_name_or_path               //设置模型权重加载路径
     ```

#### 训练结果

待补充

## 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

2024.04.25：首次发布。

# FAQ

无