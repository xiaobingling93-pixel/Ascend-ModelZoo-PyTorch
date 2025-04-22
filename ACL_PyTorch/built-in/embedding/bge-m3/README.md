# BGE-M3模型适配

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
    - [开始推理验证](#开始推理验证)
    - [性能](#性能)
  
******

# 概述
```BGE-M3```模型是BAAI General Embedding提出的先进的多语言、多功能文本Embedding模式。该模型基于Transformers Encoder，引入稀疏注意力和多向量检索，支持3种语义表示，同时还可以支持超过100种语言，最长可以处理8192序列长度，适合处理长文本。```BGE-M3```可以快速高效地生成3种不同的文本语义表示，通过语义表示间的不同组合，可以支持多种检索方式，在多语言、跨语言、长本文信息检索领域表现出色，为开发者提供了使用的工具。

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                           |   版本 | 环境准备指导                                                                                          |
  |--------------------------------------------------------------| ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                        | 25.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         |  8.1.RC1 | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                       |  3.10 | -                                                                                                     |
  | PyTorch                                                      | 2.5.1 | -                                                                                                     |
  | Ascend Extension PyTorch                                     | 2.5.1.post2 | -                                                                                                     |
  | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。  |      \ | \                                                                                                     |

# 快速上手

## 获取源码
1. 获取开源模型源码和权重（可选）
   > 如果您的设备可以方便的直接从hugging-hub下载权重和代码，则不需要执行这一步
   ```
   # git模型下载，请确保已安装git lfs
   git clone https://huggingface.co/BAAI/bge-m3
   cd bge-m3
   git reset --hard 5617a9f
   ```
   本地下载完成后的目录树如下：
   ```TEXT
    bge-m3/
    ├── colbert_linear.pt
    ├── config.json
    ├── config_sentence_transformers.json
    ├── infer.py  # 本仓库提供的自定义推理脚本
    ├── modules.json
    ├── pytorch_model.bin
    ├── sentence_bert_config.json
    ├── sentencepiece.bpe.model
    ├── sparse_linear.pt
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
    ```
2. 安装依赖
    ```SHELL
    pip3 install FlagEmbedding transformers==4.51.1 
    ```
   其他基础依赖信息可参考`requirements.txt`文件。
   
## 模型推理
### 开始推理验证
设置环境变量，执行推理命令
```SHELL
# 指定使用NPU ID，默认为0
export ASCEND_RT_VISIBLE_DEVICES=0
# 如果可以方便快速从huggingface-hub下载权重，则可以使用如下命令
# python3 infer.py --model_path=BAAI/bge-m3
python3 infer.py  # 可以使用 --model_path 指定权重路径
```
在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏E2E性能数据。如果想测试模型推理耗时，可以在 `YOUR_ENV\FlagEmbedding\inference\embedder\encoder_only\m3.py` 文件423行 `outputs = self.model(...)` 前后添加时间打点。
> 其中 YOUR_ENV 是你当前的环境路径，可以通过 ```pip show FlagEmbedding | grep Location``` 查看


### 性能
   | 模型     | 芯片             | E2E      | forward |
   |--------|----------------|----------|---------|
   | bge-m3 | Atlas 300I DUO | 137.59ms | 23.23ms |
   | bge-m3 | Atlas 800I A2  | 103.88ms | 14.71ms |