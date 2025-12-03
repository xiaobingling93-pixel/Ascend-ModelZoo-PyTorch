# ChatGLM3-6B for PyTorch

# 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [参考实现](#参考实现)
  - [适配昇腾 AI 处理器的实现](#适配昇腾-ai-处理器的实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [准备训练数据集](#准备训练数据集)
  - [准备预训练权重](#准备预训练权重)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [训练结果](#训练结果)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)


## 简介
### 模型介绍

**ChatGLM3** 是智谱AI和清华大学 KEG 实验室联合发布的对话预训练模型。ChatGLM3-6B 是 ChatGLM3
系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：

1. **更强大的基础模型：** ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base
   采用了更多样的训练数据、更充分的训练步数和更合理的训练策略。在语义、数学、推理、代码、知识等不同角度的数据集上测评显示，*
   *ChatGLM3-6B-Base 具有在 10B 以下的基础模型中最强的性能**。
2. **更完整的功能支持：** ChatGLM3-6B 采用了全新设计的Prompt格式，除正常的多轮对话外。同时原生支持工具调用（Function Call）、代码执行（Code Interpreter）和Agent 任务等复杂场景。
3. **更全面的开源序列：** 除了对话模型ChatGLM3-6B外，还开源了基础模型ChatGLM3-6B-Base
   、长文本对话模型ChatGLM3-6B-32K和进一步强化了对于长文本理解能力的ChatGLM3-6B-128K。以上所有权重对学术研究**完全开放**.
### 参考实现 ：
  ```
  url=https://github.com/THUDM/ChatGLM3
  commitID=08b01f50dccf540172d9f63e7f23b7e6de0ebc23
  ```

### 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/foundation
  ```


## 准备训练环境

### 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|软件类型	|支持版本   |
|-----|---------|
|FrameworkPTAdapter	|6.0.RC2|
|CANN	|8.0.RC2|
|昇腾NPU固件	|24.1.RC2|
|昇腾NPU驱动	|24.1.RC2|
### 安装模型环境
#### 创建conda
  ```shell
  conda create -n chatglm3-demo python=3.10
  conda activate chatglm3-demo
  ```

#### 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。
  **表 2**  版本支持表

 |Torch_Version|                  三方库依赖版本                  |
 |:-------------:|:-------------------------------------------:|
 |PyTorch 2.2.0|transformers == 4.39.3; deepspeed == 0.14.2|
  
#### 安装依赖。
  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```python
  pip install -r requirements.txt
  ```

#### 亲和优化
   优化涉及到transformers内部代码修改
- 查看transformers安装路径
  ```shell
  pip show transformers
  ```
- 计算图优化
  ```shell
  cp script/transformers_op/trainer_pt_utils.py your/path/to/transformers/trainer_pt_utils.py
  ```
- 优化器优化
  ```shell
  cp script/transformers_op/optimization.py your/path/to/transformers/optimization.py
  cp script/transformers_op/training_args.py your/path/to/transformers/training_args.py
  cp script/transformers_op/trainer.py your/path/to/transformers/trainer.py
  ```

## 准备数据集

### 准备训练数据集

  用户首先在根目录创建文件夹`data`,然后下载AdvertiseGen数据集，并将其放在`data`路径下;
- 解压AdvertiseGen.tar.gz
  ```shell
  tar -xzvf AdvertiseGen.tar.gz
  ```
- 执行`script`路径下的`process.py`脚本进行数据处理
  ```python
  python ./script/process.py
  ```
- 处理后`data`文件夹内容包括:
  ```
  ├── AdvertiseGen
  |    ├── train.json
  |    └── dev.json
  └──  AdvertiseGen_fix
        ├── dtrain.json
        └── dev.json
   
  ```

### 准备预训练权重

  用户可以自行下载ChatGLM3-6B预训练权重和配置文件(https://huggingface.co/THUDM/chatglm3-6b)，然后将这些文件放在 "models"文件夹中，**不要覆盖 `modeling_chatglm.py`文件**。
`models`文件夹内容如下：
```shell
  ├── models
      ├──config.json
      ├──configuration_chatglm.py
      ├──modeling_chatglm.py
      ├──pytorch_model-00001-of-00007.bin
      ├──pytorch_model-00002-of-00007.bin
      ├──pytorch_model-00003-of-00007.bin
      ├──pytorch_model-00004-of-00007.bin
      ├──pytorch_model-00005-of-00007.bin
      ├──pytorch_model-00006-of-00007.bin
      ├──pytorch_model-00007-of-00007.bin
      ├──pytorch_model.bin.index.json
      ├──quantization.py
      ├──tokenization_chatglm.py
      ├──tokenizer_config.json
      └──tokenizer.model
```

##  快速开始

### 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```
2. 启动全参数finetune训练

    启动8卡微调
    ```shell
    bash script/run_train_8p_glm_lora.sh
    ```
    模型训练脚本参数说明如下
    ```
    ./script/finetune_hf.py                     //脚本文件夹路径
    ./data/AdvertiseGen_fix                     //数据文件夹路径
    ./models                                    //模型权重文件夹路径
    ./configs/lora.yaml                         //配置文件夹路径
    ```
    如需要保存日志文件，请使用以下命令，并将运行脚本及日志文件名更改为自己对应的文件名
    ```bash
    nohup bash /path/to/your/sh/scripts >/path/to/your/log/file 2>&1 &
    ```

### 训练结果

**表 3**  训练结果展示表

|芯片           |卡数|模型      |Iterations| Train Steps per Second |
|---------------|---|-----------|------------|------------------------|
|竞品A          |8p|ChatGLM3-6B|1000      | 3.652                  |
|Atlas 800T A2|8p|ChatGLM3-6B|1000      | 4.826                  |





## 公网地址说明

代码涉及公网地址参考 public_address_statement.md

## 变更说明
2024.06.30：首次发布

## FAQ
暂无。
