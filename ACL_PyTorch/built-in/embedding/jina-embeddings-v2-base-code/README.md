# jina-embeddings-v2-base-code(TorchAir)-推理指导

- [jina-embeddings-v2-base-code-推理指导](#jina-embeddings-v2-base-code(TorchAir)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
    - [1 开始推理验证](#1-开始推理验证)
    - [2 性能](#2-性能)

******

# 概述
&emsp;&emsp;‌`jina-embeddings-v2-base-code` 是一款支持英语和30种常用编程语言的代码嵌入模型。它采用Bert架构和ALiBi技术，支持8192序列长度，适合处理长文档。该模型经过大规模代码数据训练，拥有1.61亿参数，可快速高效地生成嵌入。它在技术问答和代码搜索等场景表现出色，为开发者提供了实用的工具。

- 版本说明：
  ```
  url=https://huggingface.co/jinaai/jina-embeddings-v2-base-code
  commit_id=516f4ba
  model_name=jina-embeddings-v2-base-code
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 25.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  8.1.RC1 | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                          |  3.10 | -                                                                                                     |
  | PyTorch                                                         | 2.5.1 | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.5.1.post2 | -                                                                                                     |
  | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/embedding/jina-embeddings-v2-base-code
   ```

1. 获取开源模型源码和权重（可选）
   > 如果您的设备可以方便的直接从hugging-hub下载权重和代码，则不需要执行这一步
   ```
   # git模型下载，请确保已安装git lfs
   git clone https://huggingface.co/jinaai/jina-embeddings-v2-base-code
   cd jina-embeddings-v2-base-code
   git reset --hard 516f4ba

   git clone https://huggingface.co/jinaai/jina-bert-v2-qk-post-norm
   cd jina-bert-v2-qk-post-norm
   git reset --hard 3baf9e3
   cd ..
   mv jina-bert-v2-qk-post-norm bertV2qkPostNorm
   ```
   本地下载完成后的目录树如下
   ```shell
    jina-embeddings-v2-base-code
    ├── bertV2qkPostNorm    // import jina-bert-v2-qk-post-norm 会报 SyntaxError，所以修改文件名
    │   ├── config.json
    │   ├── configuration_bert.py
    │   ├── modeling_bert.py
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    ├── config.json
    ├── generation_config.json
    ├── infer.py           // 本仓库提供的自定义推理脚本
    ├── model.safetensors
    ├── modules.json
    ├── README.md
    ├── sentence_bert_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── trainer_state.json
    ├── train_results.json
    └── vocab.json
    ```


2. 安装依赖  
   ```
   pip3 install transformers==4.35.2

   ```


## 模型推理

### 1 开始推理验证

   1. 设置环境变量，执行推理命令

      ```
      # 指定使用NPU ID，默认为0
      export ASCEND_RT_VISIBLE_DEVICES=0
      # 如果可以方便快速从huggingface-hub下载权重，则可以使用如下命令
      # python3 infer.py --model_path=jinaai/jina-embeddings-v2-base-code
      python3 infer.py --model_path=./
      ```
      - --model_path: 权重路径

      在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏计算结果和E2E性能数据。如果想测试模型推理耗时，可以在 `bertV2qkPostNorm/modeling_bert.py` 文件 1179行 `token_embs = self.forward(**encoded_input)[0]` 前后添加时间打点。

### 2 性能

   |模型|芯片|E2E|forward|
   |------|------|------|---|
   |jina-embeddings-v2-base-code|Atlas 800I A2|9.4ms|5.5ms|

