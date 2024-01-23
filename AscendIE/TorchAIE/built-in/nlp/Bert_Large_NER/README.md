- [1. 概述](#1-概述)
  - [1.1. 输入输出数据](#11-输入输出数据)
- [2. 推理环境准备](#2-推理环境准备)
- [3. 快速上手](#3-快速上手)
  - [3.1. 获取模型](#31-获取模型)
  - [3.2. 准备数据集(请遵循数据集提供方要求使用)](#32-准备数据集请遵循数据集提供方要求使用)
  - [3.3. 模型推理](#33-模型推理)
- [4. 模型推理性能\&精度](#4-模型推理性能精度)



# 1. 概述

**bert-large-NER**  是一个经过微调的 BERT 模型，可用于**命名实体识别**，并为 NER 任务实现**一流的性能**。它已经过训练，可以识别四种类型的实体：位置（LOC），组织（ORG），人员（PER）和杂项（杂项）。

具体而言，此模型是一个*bert-large-cased*模型，在标准  [CoNLL-2003 命名实体识别](https://www.aclweb.org/anthology/W03-0419.pdf)数据集的英文版上进行了微调。如果要在同一数据集上使用较小的 BERT 模型进行微调，也可以使用[**基于 NER 的 BERT**](https://huggingface.co/dslim/bert-base-NER/)  版本。

* 模型权重：

  ```
  url = https://huggingface.co/dslim/bert-large-NER
  commit_id = 95c62bc0d4109bd97d0578e5ff482e6b84c2b8b9
  model_name = bert-large-NER
  ```

* 参考实现：

  ```
  git clone https://github.com/huggingface/transformers
  cd transformers
  git checkout -b v4.24.0 v4.24.0
  ```

## 1.1. 输入输出数据

- 输入数据

  | 输入数据       | 数据类型 | 大小            | 数据排布格式 |
  | -------------- | -------- | --------------- | ------------ |
  | input_ids      | int64    | batchsize x 512 | ND           |
  | attention_mask | int64    | batchsize x 512 | ND           |
  | token_type_ids | int64    | batchsize x 512 | ND           |


- 输出数据

  | 输出数据 | 大小                 | 数据类型 | 数据排布格式 |
  | -------- | -------------------- | -------- | ------------ |
  | logits   | batchsize x 512  x 9 | FLOAT32  | ND           |


# 2. 推理环境准备

- 硬件环境：310P3
  

-   该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套       | 版本             | 环境准备指导 |
| ---------- | ---------------- | ------------ |
| 固件与驱动 | 23.0.rc1         | -            |
| CANN       | 7.0.RC1.alpha003 | -            |
| Python     | 3.9.11           | -            |
| Pytorch    | 2.0.1+cpu        | -            |
| AscendIE   | 6.3.RC2          | -            |
| Torch-aie  | 6.3.RC2          | -            |



# 3. 快速上手

## 3.1. 获取模型

### 3.1.1.  获取开源模型。
  获取模型权重和配置文件，下载地址：https://huggingface.co/dslim/bert-large-NER/tree/95c62bc0d4109bd97d0578e5ff482e6b84c2b8b9

  文件结构如下：
      
    
    bert-large-NER/
    ├── README.md
    ├── config.json
    ├── flax_model.mspack
    ├── gitattributes
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tf_model.h5
    ├── tokenizer_config.json
    └── vocab.txt
    

### 3.1.2.  安装依赖
  
  ```
  pip install -r requirements.txt
  说明：如果安装了torch_npu，需要先卸载掉，避免冲突。pip uninstall torch_npu
  ```
  
## 3.2. 准备数据集(请遵循数据集提供方要求使用)

1.  获取原始数据集。

   数据集名称： **CoNll-2003**：信息抽取-命名实体识别（NER）
   下载链接：[CoNLL 2003 (English) Dataset | DeepAI](https://data.deepai.org/conll2003.zip)
   解压:
   ```
   unzip conll2003.zip
   ```
   得到如下文件夹：
   ```
   conll2003
   ├── metadata
   ├── train.txt
   ├── test.txt
   └── valid.txt
   ```
   修改离线数据集读取路径：
```commandline
vim /root/.cache/huggingface/modules/datasets_modules/datasets/conll2003/{95c62bc0d4109bd97d0578e5ff482e6b84c2b8b9...}/conll2003.py +193
downloaded_file="path/to/conll2003"
```

## 3.3. 模型推理                                                                                
### 3.3.1 模型推理
```
#拉取transformers源码
git clone https://github.com/huggingface/transformers
cd transformers
git checkout -b v4.24.0 v4.24.0

cd transformers
#将本仓库的补丁文件放到transformers/路径下
git apply patchfile.patch
pip install .
cd examples/pytorch/token-classification
python run_ner.py --model_name_or_path /path/to/bert-large-NER --dataset_name conll2003 --output_dir /tmp/test-ner --do_predict --overwrite_output_dir --no_cuda --jit_mode_eval --pad_to_max_length --max_seq_length 512 --torch_aie_enable --dataloader_drop_last --per_device_eval_batch_size 1

# 其他参数说明：max_predict_samples表示控制推理的样本数量；per_device_eval_batch_size设置推理的batch_size大小。
```

# 4. 模型推理性能&精度

## 4.1. 性能对比

| Batch Size |  om推理(QPS)   | torch-aie推理(QPS) | torch-aie/om |
| :--------: |:------------:|:----------------:|:------------:|
|     1      |   63.3062    |     43.5007      |    0.6871    |
|     4      |   70.6858    |     42.2668      |    0.5980    |
|     8      | 74.5971   |     37.7312      |    0.5058    |
|     16      | 73.7290 |     35.8528      |    0.4863    |
|     32     | 73.6084  |     35.1328      |    0.4773    |
|     64     | 71.1308  |     36.2688      |    0.5099    |

> 性能有改进空间，待通过aie接入bert优化pass。

## 4.2. 精度对比

|      模型      | Batch Size | om推理 | torch-aie推理 |
| :------------: | :--------: | :----: | :-----------: |
| bert_large_NER |     1      | 90.74% |    90.90%     |
| bert_large_NER |     4      | 90.74% |    90.91%     |
| bert_large_NER |     8      | 90.74% |    90.92%     |
| bert_large_NER |     16      | 90.74% |    90.89%     |
| bert_large_NER |     32     | 90.74% |    90.88%     |
| bert_large_NER |     64     | 90.74% |    90.85%     |
