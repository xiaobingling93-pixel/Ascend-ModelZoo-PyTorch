# CogVLM for Pytorch
# 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [CogVLM](#CogVLM)
  - [准备训练环境](#准备训练环境)
  - [准备数据集](#准备数据集)
  - [快速开始](#快速开始)
  - [微调任务](#微调任务)
  - [推理任务](#推理任务)
- [公网地址变更说明](#公网地址变更说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)



# 简介
## 模型介绍
CogVLM is VISUAL EXPERT FOR LARGE LANGUAGE 是一个多模态视觉-文本模型，它强调“视觉优先”，使用11B参数建模图像特征，多于文本的7B参数量。该模型包含ViT编码器、MLP适配器、预训练大语言模型和视觉专家模块，通过深度整合语言和视觉信息，提升了跨模态任务的性能。在多个基准测试中，CogVLM展现出领先或次领先的性能，显示出其在视觉理解研究和工业应用中的巨大潜力。

官方仓：https://github.com/THUDM/CogVLM

说明：本仓代码仅为适配官方仓脚本，执行训练与推理需要在官方仓cogvlm项目路径下进行。

## 支持任务列表
本仓已支持以下模型任务类型。

| 模型         | 模型大小 | 任务类型       | 是否支持  |
|------------|------|------------| ------------ |
| CogVLM |   base-224   | 微调         | ✅   |

## 代码实现
- 参考实现
  ```
  CogVLM仓: https://github.com/THUDM/CogVLM
  commit id: eb2367f54b95da2ee64f996305ab1baa45df7479
  ```
- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation/CogVLM
  ```
  
# CogVLM

## 准备训练环境
### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|        软件类型        |   支持版本   |
|:------------------:|:--------:|
| FrameworkPTAdapter | 6.0.RC2  |
|        CANN        | 8.0.RC2  |
|      昇腾NPU固件       | 24.1.RC2 |
|      昇腾NPU驱动       | 24.1.RC2 |

### 安装模型环境

**表 2**  三方库版本支持表

|    三方库    |  支持版本  |
|:---------:|:------:|
|  PyTorch  | 2.1.0  |

1) 安装模型对应PyTorch版本需要的依赖, 需要先安装PTA包。
2) 下载model_zoo下面的CogVLM相关文件，并依赖该路径下的requirements.txt进行三方件安装。
```shell
pip install -r requirements.txt
```
3) 下载并安装en_core_web_sm-any-py3-none-any.whl[下载](https://huggingface.co/spacy/en_core_web_sm/tree/main)，
en_core_web_sm是spaCy 自然语言处理（NLP）工具库中的一种语言模型，专为英语设计。
注：若安装过程出现问题，请参考FAQ-1
### 准备数据集

1) 微调数据集:
训练与评估所使用的数据集为Captcha Images dataset(验证码数据集)[下载](https://www.kaggle.com/datasets/aadhavvignesh/captcha-images)，该数据集是官网提供的一个预训练数据集。

2) label数据:
数据的label信息为图像的文件名。

下载完后文件夹结构如下所示：

```text
archive
├── 004rVO6G09.jpg
├── 00949IT0LT.jpg
├── 00bAQwhAZU.jpg
├── 01S19jY65H.jpg
...
```
#### 微调数据预处理
数据下载完成后，需要对其进行数据集划分，train/validation/test的划分比例为80/5/15，在官网utils/split_dataset.py中指定源文件路径，如下面代码中的"archive"路径：
```python
all_files = find_all_files('archive')
```
执行数据划分操作，如下命令：
```shell[dataset.py](cogvlm_utils%2Fdataset.py)
python utils/split_dataset.py
```
划分后会生成train/valid/test文件，文件中分别包含划分后的图像。
```text
archive_split
├── test
├── train
├── valid
```

### 获取预训练权重

1) 官方提供微调权重cogvlm-base-224[下载](https://huggingface.co/THUDM/CogVLM/tree/main)。

2) 分词器权重[下载](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main)。

## 快速开始

### 微调任务
主要提供基于Captcha Images数据集微调的8卡训练脚本。
#### 模型适配

1) 模型文件替换:

替换fintune_cogvlm_change.sh脚本中开头三个路径，model_zoo路径、Cogvlm路径和sat路径为实际路径，执行脚本进行替换，如下命令：
```shell
sh fintune_cogvlm_change.sh
```
#### 执行微调
1) finetune_demo/finetune_cogvlm_base_224.sh文件夹下，修改文件中微调权重路径、分词器权重路径和数据集路径（train_data和valid_data）为实际路径。

2) 执行训练，如下命令：
```
cd finetune_demo
bash finetune_cogvlm_base_224.sh
```

#### 训练结果
##### 说明
模型中包含多种随机问题，会影响loss曲线和下游任务，用户可根据需要自行修改，本代码不做更换：
1) Cogvlm项目路径的utils/utils/dataset.py 中load_data中的all_files是无序的，可以通过以下方式固定：
```python
all_files.sort()
```
2) 模型本身有确定性问题
3) SwissArmyTransformer三方件的sat/model/transformer.py文件中的BaseTransformer下embedding_dropout_prob、attention_dropout_prob和output_dropout_prob三个dropout不为0
##### 精度

基于Captcha Images数据集训练800步、1600步和2000步验证下游任务，由于模型本身有随机性问题，因此下游任务在评估数据上略有波动：

|    芯片    |  800  | 1600   | 2000  | 
|:--------:|:-----:|--------|-------|
|   GPU    |  95%  | 96.25% | 97.5% |
| Atlas A2 |  95%  | 95%    | 97.5% | 

##### 性能


|    芯片    | 卡数 | samples/s | batch_size | AMP_Type | Torch_Version |
|:--------:| :----: |:---------:|:----------:|:--------:| :-----------: |
|   GPU    |   8p   |    335    |     4      |   bf16   |      2.1      |
| Atlas A2 |   8p   |    265    |     4      |   bf16   |      2.1      |

#### 微调后推理
1) finetune_demo/eval_cogvlm_base_224.sh文件夹下，修改文件中微调后权重路径、分词器路径和数据集路径(test_data)为实际路径。
2) 执行推理，如下命令：
```shell
bash eval_cogvlm_base_224.sh
```

### 推理任务
该处的推理为官方hf权重的推理方式，与微调后推理方式略有区别，用户可根据实际情况判断是否执行。
#### 推理前准备
1) 预训练权重cogvlm-base-224-hf[下载](https://huggingface.co/THUDM/cogvlm-base-224-hf)。

2) 替换inference_cogvlm_change.sh脚本中model_zoo路径、Cogvlm路径和HF权重路径为实际路径,执行脚本进行替换。
```shell
sh inference_cogvlm_change.sh
```
3) finetune_demo/inference.py文件夹下，并根据实际路径修改推理权重路径、分词器权重路径和图片路径
#### 启动推理
```shell
cd finetune_demo
source ./env_npu.sh
python inference.py
```

# 公网地址变更说明
暂无。

# 变更说明
2024.03.30：CogVLM bf16微调任务首次发布。


# FAQ

1. 安装en_core_web_sm-any-py3-none-any.whl报错 “Invalid requirement: en_core_web_sm==any”。

   解决方案：修改 en_core_web_sm-any-py3-none-any.whl 包名为 en_core_web_sm-3.7.1-py3-none-any.whl 后重新安装。