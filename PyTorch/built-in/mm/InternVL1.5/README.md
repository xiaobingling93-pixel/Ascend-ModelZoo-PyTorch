
# InternVL1.5 for PyTorch


# 目录

- [InternVL1.5 for PyTorch](#InternVL1.5-for-pytorch)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [InternVL1.5（在研版本）](#InternVL1.5在研版本)
  - [准备训练环境](#准备训练环境)
    - [安装模型环境](#安装模型环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
      - [准备训练数据集](#准备训练数据集)
    - [准备预训练模型权重](#准备预训练模型权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [单机训练任务](#单机训练任务)
        - [开始训练](#开始训练)
      - [多机训练任务](#多机训练任务)
        - [开始训练（torchrun）](#开始训练（torchrun）)
        - [开始训练（deepspeed）](#开始训练（deepspeed）)
    - [推理任务](#推理任务)
      - [开始推理](#开始推理)
  - [训练结果展示](#训练结果展示)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
  - [变更](#变更)
- [FAQ](#faq)

# 简介

## 模型介绍

上海AI Lab 推出的 InternVL 1.5 是一款开源的多模态大语言模型 (MLLM)，旨在弥合开源模型和专有商业模型在多模态理解方面的能力差距。\
论文称，InternVL 1.5 在四个特定基准测试中超越了 Grok-1.5V、GPT-4V、Claude-3 Opus 和 Gemini Pro 1.5 等领先的闭源模型，特别是在与 OCR 相关的数据集中。

InternVL 的三个改进：\
（1）强视觉编码器：为大规模视觉基础模型 InternViT-6B 探索了一种持续学习策略，提高了其视觉理解能力，并使其可以在不同的LLM中迁移和重用。\
（2）动态高分辨率：根据输入图像的长宽比和分辨率，将图像划分为1到40个448×448像素的图块，最高支持4K分辨率输入。\
（3）高质量的双语数据集：收集了高质量的双语数据集，涵盖常见场景、文档图像，并用英文和中文问答对进行注释，显着提高了 OCR 和中文相关任务的性能。

## 支持任务列表

本仓已经支持以下模型任务类型

  **表 1**  模型任务类型支持表
|     模型      | 任务列表 | 是否支持 |
|:-----------:|:----:|:-----:|
| InternVL-Chat-V1-5  | 微调训练 | ✔ |
| InternVL-Chat-V1-5  | 在线推理 | ✔ |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/OpenGVLab/InternVL.git
  commit_id=c62fa4f7c850165d7386bdc48ac6bc5a6fab0864
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/InternVL1.5
  ```


# InternVL1.5（在研版本）

## 准备训练环境

### 安装模型环境


  **表 2**  三方库版本支持表

  |     三方库     |  支持版本  |
  |:-----------:|:------:|
  |   PyTorch   | 2.1.0  |
  | TorchVision | 0.16.0 |

   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。
   ```shell
   cd InternVL1.5
   pip install -e internvl_chat  # 安装本地代码仓
   pip install -r requirements.txt  # 安装依赖包
   ```
   注：由于decord官方未发布ARM平台安装包，需要源码编译安装，如不涉及视频编解码可按照FAQ注释相关代码。

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表4中软件版本。
                
  
  **表 3**  昇腾软件版本支持表

  | 软件类型   |   支持版本   |
  | :--------: |:--------:|
  | FrameworkPTAdapter |   在研版本   |
  | CANN |   在研版本   |
  | 昇腾NPU固件 |   在研版本   | 
  | 昇腾NPU驱动 | 在研版本 |


### 准备数据集

#### 准备训练数据集

1. 请参考[InternVL官网](https://internvl.readthedocs.io/en/latest/internvl1.2/reproduce.html#training-datasets-preparation)准备微调数据集。

2. 将数据集全部解压，并放在`internvl_chat/playground`目录下，目录结构如下：
  ```shell
  internvl_chat/playground/
  ├── data
  │   ├── ai2d
  │   ├── chartqa
  │   ├── coco
  │   ...
  │   └── wikiart
  └── opensource
      ├── ai2d_train_12k.jsonl
      ├── chartqa_train_18k.jsonl
      ...
      └── synthdog_en.jsonl
  ```

### 准备预训练模型权重

1. 联网情况下，预训练模型权重会自动下载。

2. 无网络时，用户可访问[huggingface模型官网](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)自行下载。

3. 将下载好的InternVL-Chat-V1-5模型权重放在本工程目录下的`./internvl_chat/pretrained/`目录下，目录结构如下：
  ```shell
  internvl_chat/pretrained/InternVL-Chat-V1-5
  ├── config.json
  ├── model-00001-of-00011.safetensors
  ├── model-00002-of-00011.safetensors
  ...
  ├── model-000011-of-00011.safetensors
  ├── tokenizer_config.json
  └── tokenizer.model
  ```


## 快速开始

### 训练任务

本任务主要以全参微调为主，展示训练任务，包含单机多卡和多机多卡的训练。

#### 单机训练任务

##### 开始训练

1. 进入解压后的源码包根目录。

  ```shell
  cd /path/InternVL1.5
  ```

2. 准备训练数据。

  按照准备数据集及准备预训练模型权重章节，准备对应数据集和权重文件，放在模型文件夹下。

3. 运行训练脚本。

  用户可以按照自己训练需要进行参数配置，以下给出单机8卡的一种训练示例。(如训练时OOM可参考FAQ修改模型层数)
  ```shell
  # 混合精度BF16，8卡训练
  bash test/train_full_8p_bf16.sh
  # 混合精度BF16，8卡性能测试
  bash test/train_performance_8p_bf16.sh
  ```

#### 多机训练任务

##### 开始训练（torchrun）（推荐）

1.2.步骤同上

3. 运行训练脚本。

  用户可以按照自己训练需要进行参数配置，以下给出双机16卡的一种训练示例。
  ```shell
  # 主节点
  MASTER_ADDR={主节点IP} NODE_RANK=0 bash test/train_full_16p_bf16_torchrun.sh
  # 从节点
  MASTER_ADDR={主节点IP} NODE_RANK=1 bash test/train_full_16p_bf16_torchrun.sh
  ```

##### 开始训练（deepspeed）

1.2.步骤同上

3. 配置hostfile

  在test目录下新建文件hostfile，每行内容为“{主机名} slots={节点上卡数}”，示例如下：
  ```shell
  192.168.1.2 slots=8 #（第一行为主节点）
  192.168.1.3 slots=8
  ```
  注：文件末尾不可有空行

4. 配置节点间免密

  在主节点上执行ssh-copy-id {从节点IP}，配置主节点免密登录所有从节点

5. 运行训练脚本

  在主节点上运行脚本。用户可以按照自己训练需要进行参数配置，以下给出双机16卡的一种训练示例。
  ```shell
  bash test/train_full_16p_bf16_deepspeed.sh
  ```


### 推理任务

本任务主要展示推理任务，包括单机在线推理。

#### 开始推理

1. 进入解压后的源码包根目录。

  ```
  cd path/InternVL1.5
  ```

2. 修改脚本参数。

  按照自己需要修改internvl_chat/internvl/train/internvl_chat_inference.py中的MODEL_PATH和IMAGE_PATH，并修改相关任务。

3. 运行推理的脚本。

  ```shell
  python internvl_chat/internvl/train/internvl_chat_inference.py
  ```

# 训练结果展示

**表 4**  训练结果展示表

| NAME | 卡数 | Train Loss(5000steps) | FPS(Samples Per Second) | Train Steps Per Second |
| :-----: | :----: |  :---:  | :----: | :----: |
| Atlas 800T A2 | 8 | 1.9068（5层ViT+12层LLM，关闭dropout） | 4.952 | 0.145 |
| 竞品 | 8 | 1.9071（5层ViT+12层LLM，关闭dropout） | 5.151 | 0.161 |

# 公网地址说明

代码涉及公网地址参考[public_address_statement.md]

# 变更说明

## 变更

2024.09.1：InternVL1.5 bf16训练和推理任务首次发布。

# FAQ

## 1. 注释decord相关代码

注释internvl_chat/internvl/train/dataset.py第7行：
  ```python
  from decord import VideoReader
  ```

## 2.修改模型层数

修改模型权重中的config.json文件，例如internvl_chat/pretrained/InternVL-Chat-V1-5/config.json。

在文件的第66行llm_config中修改LLM层数配置，例如12:
```json
"num_hidden_layers": 48,
```

在文件的第132行vision_config中修改ViT层数配置，例如5:
```json
"num_hidden_layers": 45,
```
