
# PIDM for PyTorch
# 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装模型环境](#安装模型环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [准备数据集](#准备数据集)
- [快速开始](#快速开始)
  - [模型训练](#模型训练)
  - [结果展示](#结果展示)
  - [模型推理](#模型推理)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#faq)



# 简介

## 模型介绍
PIDM 是一种基于扩散模型的人物图像生成网络。模型利用扩散模型和人物姿态信息，进行姿态迁移训练。可以用于多种任务，包括服装迁移、风格混合和行人重识别。

## 支持任务列表
本仓已经支持以下模型任务类型：

|      模型      | 任务列表 | 是否支持 |
|:------------:|:----:|:-----:|
| PIDM |  训练  | ✔ |
| PIDM |  推理  | ✔ |

## 代码实现
- 参考实现：

  ```
  url=https://github.com/ankanbhunia/PIDM
  commit_id=e4f1d88
  ```

- 适配昇腾AI处理器的实现：
  ```shell
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/PIDM
  ```

# 准备训练环境

## 安装模型环境

- 下载代码：
  ```shell
  git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
  cd PyTorch/built-in/mlm/PIDM
  ```

- 创建Python环境并且安装Python三方包：
  ```shell
  conda create -n pidm python=3.8 -y
  conda activate pidm
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
- 数据集准备参考：（https://github.com/ankanbhunia/PIDM/blob/main/README.md 中的数据准备章节）。
- 在训练脚本中（train_pidm_multi_npu.sh（单机多卡）、 train_pidm_npu_multi_node.sh（多机多卡）通过指定dataset_path的value指定数据集路径。
    
# 快速开始

## 模型训练

1. 训练脚本位置位于mlm/PIDM目录，提供了train_pidm_multi_npu.sh（单机多卡）、 train_pidm_npu_multi_node.sh（多机多卡）两个脚本。 需要根据真实值配置cann的set_env.sh路径、数据集路径。当脚本中传入--use_bf16参数时，使能BF16混合精度，不传则为Float32精度。

2. 运行训练脚本，下面以单机多卡示例：

    ```shell
    bash train_pidm_multi_npu.sh 
    ```
## 结果展示

**表 2**  训练结果展示：

|         芯片         | 卡数 |  FPS  | batch_size | AMP_Type | Torch_Version |
|:------------------:|:---:|:-----:|:----------:|:--------:|:---:|
|        竞品A         | 8p | 18.16 |     14     |   bf16   | 2.1 |
| Atlas 900 A2 PODc  | 8p | 14.87 |     14     |   bf16   | 2.1 |
|        竞品A         | 8p | 16.35 |     14     |   fp32   | 2.1 |
| Atlas 900 A2 PODc | 8p | 14.70 |     14     |   fp32   | 2.1 |

1
## 模型推理
- 准备推理相关数据：
从 [https://github.com/ankanbhunia/PIDM/tree/main/data](https://github.com/ankanbhunia/PIDM/tree/main/data) 链接下载
deepfashion_256x256、fashion e-commerce images目录及其下属所有文件，放至PTIDM/data目录。
从 [https://github.com/ankanbhunia/PIDM/tree/main](https://github.com/ankanbhunia/PIDM/tree/main) 链接下载test.jpg文件，放至PIDM目录。
- 推理脚本位于模型根目录下，提供了predict.py脚本，其中的模型文件路径等，按照实际填写。
- 推理前加载环境变量
   ```
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python predict.py  # fp32 推理
  python predict.py --use_fp16 # fp16 推理
   ```

   脚本执行完会在当前路径下生成output.png。

# 公网地址说明

代码涉及公网地址参考 [public_address_statement.md](public_address_statement.md)


# 变更说明
2024.08.30: 首次发布。

# FAQ
无
