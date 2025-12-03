
# PLLaVA for PyTorch
# 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装模型环境](#安装模型环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [准备数据集](#准备数据集)
  - [获取预训练模型](#获取预训练模型)
- [快速开始](#快速开始)
  - [模型训练](#模型训练)
  - [结果展示](#结果展示)
  - [模型推理](#模型推理)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#faq)



# 简介

## 模型介绍

PLLaVA是一种新颖的端到端训练的大型多模态模型，它结合了视觉编码器和Vicuna，用于通用的视觉和语言理解，实现了令人印象深刻的聊天能力，在科学问答（Science QA）上达到了新的高度。

## 支持任务列表
本仓已经支持以下模型任务类型：

|      模型      | 任务列表 | 是否支持 |
|:------------:|:----:|:-----:|
| LLaVA 1.6 7B |  训练  | ✔ |
| LLaVA 1.6 7B |  推理  | ✔ |

## 代码实现
- 参考实现：

  ```
  url=https://github.com/magic-research/PLLaVA
  commit_id=6f49fd2
  ```

- 适配昇腾AI处理器的实现：
  ```shell
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mm/PLLaVA
  ```

# 准备训练环境

## 安装模型环境

- 下载代码：
  ```shell
  git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
  cd PyTorch/built-in/mm/PLLaVA
  ```

- 创建Python环境并且安装Python三方包：
  ```shell
  conda create -n pllava python=3.10 -y
  conda activate pllava
  pip install --upgrade pip  # enable PEP 660 support
  pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu  #For X86
  pip3 install torch==2.1.0  #For Aarch64
  pip install -r requirements.txt
  ```
  
## 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表4中软件版本。
                
  
  **表 4**  昇腾软件版本支持表

  | 软件类型   |   支持版本   |
  | :--------: |:--------:|
  | FrameworkPTAdapter |   在研版本   |
  | CANN |   在研版本   |
  | 昇腾NPU固件 |   在研版本   | 
  | 昇腾NPU驱动 | 在研版本 |


## 准备数据集

- json文件下载路径参考： (https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)。 
- 视频文件下载参考：（https://github.com/magic-research/PLLaVA/blob/main/README.md 中的数据准备章节）。
- 数据集结构如下所示：
   ```
    dataset/VideoChat2-IT/video/reasoning/clever_qa
      ├── train.json
    
    dataset/video_all
      ├── xxx.mp4

- 在训练脚本中（train_pllava_single_npu.sh（单卡）、 train_pllava_multi_npu.sh（单机多卡）、train_pllava_npu_multi_node.sh（多机多卡））通过指定train_corpus的value，在 tasks/train/instruction_data.py中获取具体的json路径和视频路径。

## 获取预训练模型

- 联网情况下，预训练模型会自动下载。

- 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：
   参考 https://github.com/magic-research/PLLaVA/blob/main/README.md 中的模型下载准备章节。
   在训练脚本中，需要指定模型存储的绝对路径。
    
# 快速开始

## 模型训练

1. 训练脚本位置位于scripts目录，提供了train_pllava_single_npu.sh（单卡）、 train_pllava_multi_npu.sh（单机多卡）、train_pllava_npu_multi_node.sh（多机多卡）三个脚本。 需要根据真实值配置cann的set_env.sh路径、数据集路径、权重的路径。

2. 运行训练脚本，下面以单机单卡示例：

    ```shell
    bash scripts/train_pllava_single_npu.sh 
    ```
   训练完成后，权重文件保存在参数`--output_dir`路径下。
## 结果展示

**表 2**  训练结果展示：

|         芯片          | 卡数 | second per step | batch_size | AMP_Type | Torch_Version |
|:-------------------:|:---:|:---------------:|:----------:|:--------:|:---:|
|         竞品A         | 8p |     0.9352s     |     1      |   bf16   | 2.1 |
| Atlas 200T A2 Box16 | 8p |     0.8411s     |     1      |   bf16   | 2.1 |
|         竞品A         | 8p |     1.0760s     |     1      |   fp32   | 2.1 |
| Atlas 200T A2 Box16 | 8p |     0.9347s     |     1      |   fp32   | 2.1 |


## 模型推理
训练脚本位置位于scripts目录下，提供了eval_single.sh脚本，其中的cann的set_env.sh路径、视频文件路径、模型文件路径、权重文件路径等，按照实际填写。

   ```
  bash scripts/eval_single.sh
   ```
脚本执行中，会让用户输入问题，再根据问题返回答案。

# 公网地址说明

代码涉及公网地址参考 [public_address_statement.md](public_address_statement.md)


# 变更说明
2024.08.09: 首次发布。

# FAQ
无
