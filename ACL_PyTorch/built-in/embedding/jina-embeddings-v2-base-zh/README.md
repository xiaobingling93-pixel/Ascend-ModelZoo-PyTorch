# jina-embeddings-v2-base-zh(TorchAir)-推理指导

- [jina-embeddings-v2-base-zh-推理指导](#jina-embeddings-v2-base-zh(TorchAir)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
    - [1 开始推理验证](#1-开始推理验证)
    - [2 性能](#2-性能)

******

# 概述
&emsp;&emsp;‌`jina-embeddings-v2-base-zh` 基于Transformer架构的中文文本向量模型，支持句子相似度计算、文本分类、检索和重排序功能。在MTEB中文基准测试中完成了医疗问答、电商等领域的评估，支持中英双语处理，采用Apache-2.0开源许可证。

- 版本说明：
  ```
  url=https://huggingface.co/jinaai/jina-embeddings-v2-base-zh
  commit_id=c1ff908
  model_name=jina-embeddings-v2-base-zh
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
  | Ascend Extension PyTorch                                        | 2.5.1 | -                                                                                                     |
  | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/embedding/jina-embeddings-v2-base-zh
   ```

1. 获取开源模型源码和权重（可选）
   > 如果您的设备可以方便的直接从hugging-hub下载权重和代码，则不需要执行这一步
   ```
   # git模型下载，请确保已安装git lfs
   git clone https://huggingface.co/jinaai/jina-embeddings-v2-base-zh
   cd jina-embeddings-v2-base-zh
   git reset --hard c1ff908

   git clone https://huggingface.co/jinaai/jina-bert-implementation
   cd jina-bert-implementation
   git reset --hard f3ec4cf
   cd ..
   mv jina-bert-implementation jinaBertImplementation
   ```
   本地下载完成后的目录树如下
   ```shell
    jina-embeddings-v2-base-zh
    ├── jinaBertImplementation    // import jina-bert-implementation 会报 SyntaxError，所以修改文件名
    │   ├── README.md
    │   ├── configuration_bert.py
    │   ├── modeling_bert.py
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    ├── config.json
    ├── config_sentence_transformers.json
    ├── merges.txt
    ├── infer.py           // 本仓库提供的自定义推理脚本
    ├── model.safetensors
    ├── modules.json
    ├── README.md
    ├── sentence_bert_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json
    ```


2. 安装依赖  
   ```
   pip3 install transformers==4.35.2 torch==2.5.1 torch_npu==2.5.1 protobuf numpy==1.26.4 decorator attrs psutil scipy

   ```


## 模型推理

### 1 开始推理验证

   1. 设置环境变量，执行推理命令

      ```
      # 指定使用NPU ID，默认为0
      export ASCEND_RT_VISIBLE_DEVICES=0
      # 如果可以方便快速从huggingface-hub下载权重，则可以使用如下命令
      # python3 infer.py --model_path=jinaai/jina-embeddings-v2-base-zh
      python3 infer.py --model_path=./
      ```
      - --model_path: 权重路径

      在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏计算结果和E2E性能数据。如果想测试模型推理耗时，可以在 `bertV2qkPostNorm/modeling_bert.py` 文件 1179行 `token_embs = self.forward(**encoded_input)[0]` 前后添加时间打点。

### 2 性能

   |模型|芯片|E2E|forward|
   |------|------|------|---|
   |jina-embeddings-v2-base-zh|Atlas 800I A2|7.9ms|4.4ms|

