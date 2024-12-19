
# LLaVA for PyTorch

**注意**： 本仓库中LLaVA模型将不再进行维护，请使用[MindSpeed-MM](https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/llava1.5)


# 目录
- [LLaVA](#llava-for-pytorch)
  - [概述](#概述)
  - [准备训练环境](#准备训练环境)
    - [创建Python环境](#创建python环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [模型训练](#模型训练)
    - [结果展示](#结果展示)
    - [模型评估](#模型评估)
    - [模型推理](#模型推理)
  - [公网地址说明](#公网地址说明)
  - [变更说明](#变更说明)
  - [FQA](#faq)



## 概述

### 模型介绍

LLaVA是一种新颖的端到端训练的大型多模态模型，它结合了视觉编码器和Vicuna，用于通用的视觉和语言理解，实现了令人印象深刻的聊天能力，在科学问答（Science QA）上达到了新的高度。

### 支持任务列表
本仓已经支持以下模型任务类型

|      模型      | 任务列表 | 是否支持 |
|:------------:|:----:|:-----:|
| LLaVA 1.5 7B |  训练  | ✔ |
| LLaVA 1.5 7B |  推理  | ✔ |
| LLaVA 1.5 7B |  评估  | ✔ |

### 代码实现
- 参考实现：

  ```
  url=https://github.com/haotian-liu/LLaVA.git
  commit_id=3e337ad269da3245643a2724a1d694b5839c37f9
  ```

- 适配昇腾AI处理器的实现：
  ```shell
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/LLaVA
  ```

## 准备训练环境

### 创建Python环境

- git clone 远程仓
  ```shell
  git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
  cd PyTorch/built-in/mlm/LLaVA
  ```

- 创建Python环境并且安装Python三方包
  ```shell
  conda create -n llava python=3.10 -y
  conda activate llava
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  pip install -e ".[train]"
  pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu  #For X86
  pip3 install torch==2.1.0  #For Aarch64
  pip3 install accelerate==0.28.0 decorator==5.1.1 scipy==1.13.0 attrs==23.2.0 openpyxl
  ```
  **如果不使用wandb需要卸载wandb，否则程序会报错**
  ```shell
  pip uninstall wandb
  ```
- 环境准备指导

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

    **表 1**  昇腾软件版本支持表

  |     软件类型     |   支持版本   |
  |:-----------:|:--------:|
  | FrameworkPTAdapter  |   在研版本   |
  | CANN | 在研版本 |
   | 昇腾NPU固件 | 在研版本 |
   | 昇腾NPU驱动 | 在研版本 |


### 准备数据集

- 需要自行下载llava_v1_5_mix665k.json指令微调数据集，以及图片数据集，涉及到的数据集结构如下所示：
   ```
    playground/data
      ├── llava_v1_5_mix665k.json
      ├── coco
      │   └── train2017
      ├── gqa
      │   └── images
      ├── ocr_vqa
      │   └── images
      ├── textvqa
      │   └── train_images
      └── vg
          ├── VG_100K
          └── VG_100K_2
   ```
  需要将这五个数据集放置到同一个文件夹下，数据集来源请参考 https://github.com/haotian-liu/LLaVA/blob/main/README.md 中的数据准备章节。
### 准备预训练权重

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：
    ```shell
    lmsys/vicuna-7b-v1.5
    openai/clip-vit-large-patch14-336
    liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
    ```
## 快速开始

### 模型训练

1. 训练脚本位置位于scripts/v1_5/finetune_npu.sh，需要手动将数据集，权重的路径传入到相应参数上，路径仅供参考，请用户根据实际情况修改。
   ```shell
    --model_name_or_path lmsys/vicuna-7b-v1.5  # vicuna权重路径
    --data_path ./playground/data/llava_v1_5_mix665k.json # 指令微调数据的路径
    --image_folder ./playground/data # 图片数据集的路径，路径下包含五个数据集
    --vision_tower openai/clip-vit-large-patch14-336 # clip模型路径
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin # mlp adapter路径
   ```

2. 运行训练脚本，该模型支持单机8卡训练。

    ```shell
    bash scripts/v1_5/finetune_npu.sh # 8卡精度及性能 bf16
    ```
   训练完成后，权重文件保存在参数`--output_dir`路径下。
### 结果展示

**表 2**  训练结果展示

| 芯片 | 卡数 | samples per second | batch_size | AMP_Type | Torch_Version |
|:---:|:---:|:------------------:|:----------:|:---:|:---:|
| GPU | 8p |       18.62        |     16     | bf16 | 2.1 |
| Atlas A2 | 8p |       20.13        |     16     | bf16 | 2.1 |

### 模型评估

 评估脚本位于scripts/v1_5/eval下面，这里以textvqa任务为例，测试之前需要准备TextVQA_0.5.1_val.json和train_val_images.zip数据集，解压并传到textvqa.sh相关参数上面，执行textvqa.sh脚本即可进行评估。
  ```
  --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
  --image-folder ./playground/data/eval/textvqa/train_images \
  ```

### 模型推理

 推理任务需要传递训练好的模型以及图片到下面的脚本上，执行下面命令即可进行推理。
   ```
  python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
   ```

## 公网地址说明

代码涉及公网地址参考 public_address_statement.md


## 变更说明
2024.05.09: 首次发布

2024.05.20: 添加NPU适配代码

## FAQ
无
