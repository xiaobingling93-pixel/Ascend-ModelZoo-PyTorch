# Grounding DINO for Pytorch

# 目录

-   [简介](#简介)
    -   [模型介绍](#模型介绍)
    -   [支持任务列表](#支持任务列表)
    -   [代码实现](#代码实现)
-   [Grounding-dino](#Grounding-dino)
    -   [准备训练环境](#准备训练环境)
    -   [准备数据集](#准备数据集)
    -   [快速开始](#快速开始)
        -   [微调任务](#微调任务)
        -   [推理任务](#推理任务)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)


# 简介

## 模型介绍

Grounding DINO是一个开放集目标检测模型，可以根据输入文本检测文本对应的目标。该模型通过将语言模型引入闭集检测器以进行开放集概念泛化，可以检测带有人类输入的任意对象。本仓库主要将Grounding DINO模型迁移到了昇腾NPU上，并进行了性能优化。

官方仓：https://github.com/IDEA-Research/GroundingDINO

## 支持任务列表

本仓已支持以下模型任务类型。

| 模型             | 任务类型 | 是否支持  |
|----------------|------| ------------ |
| Grounding-dino | 微调   | ✅   |



## 代码实现

- 参考实现：

  ```
  MMDetection仓：https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino
  commit_id=2390ebc32384512477c6c1dd51a452a71f45e908
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```

# Grounding-dino
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
|        Mx_Driving         | 6.0.RC2  |

### 安装模型环境

**表 2**  三方库版本支持表

|    三方库    |  支持版本  |
|:---------:|:------:|
|  PyTorch  | 2.1.0  |


### 脚本
#### 搭建环境
 ```shell
 # python3.8
 conda create -n test python=3.8
 conda activate test
 
 # 安装 torch 和 torch_npu 
 pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
 pip install torch_npu-2.1.0.XXX-cp38-cp38m-linux_aarch64.whl

 # 修改 ascend-toolkit 路径
 source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
 pip install torchvision==0.16.0
 pip install opencv-python
 pip install opencv-python-headless
 pip install mmengine==0.10.3
 pip install jsonlines
 pip install nltk
 
 # 安装 mmcv
 git clone https://github.com/open-mmlab/mmcv
 cd mmcv/mmcv
 vim version.py  # 此处需要把第二行的__version__ = '2.2.0'改成'2.1.0'，然后保存退出
 cd ..
 pip install -r requirements.txt
 MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext 
 MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
 cd ..
 git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 mmdetection
 cd mmdetection
 pip install -r requirements.txt
 mkdir -p weights/bert weights/nltk_data weights/g_dino_model refcoco/mdetr_annotations
 ```
- 安装mxDriving加速库，安装方法参考[原仓](https://gitee.com/ascend/mxDriving)，安装后根据原仓**快速上手**章节将source环境变量命令添加在groundingdino_npu/env_npu.sh中。

#### 将groundingdino_npu目录放在mmdetection目录下

```shell
mv ../groundingdino_npu ./
pip install -r groundingdino_npu/requirements.txt
```

### 准备数据集

#### 微调数据集
训练与评估所使用的数据集为refcoco数据集，需要用户在mmdetection目录下创建refcoco文件夹。

1. 用户需自行获取COCO2014数据集，将训练集命名为train2014

   将train2014放置在./refcoco目录下
   
2. refcoco对应的标注数据

   参考[链接](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations)
   
   获取文件：finetune_refcoco_val.json、finetune_refcoco_testA.json、finetune_refcoco_testB.json、finetune_refcoco_train.json

   在refcoco目录下创建文件夹mdetr_annotations，并将这四个json文件放置在./refcoco/mdetr_annotations目录下
   

准备好后文件夹结构如下所示：

```text

├refcoco
├── train2014
│   ├── xxx.jpg
│   ├── ...
├── mdetr_annotations
│   ├── finetune_refcoco_val.json
│   ├── finetune_refcoco_testA.json
│   ├── finetune_refcoco_testB.json
│   ├── finetune_refcoco_train.json
```

然后需要使用以下命令将train.json文件转换为所需的 ODVG 格式：
```shell
python tools/dataset_converters/refcoco2odvg.py refcoco/mdetr_annotations
```
  > **说明：** 
  > 以上命令会在mdetr_annotations文件夹下生成finetune_refcoco_train_vg.json文件，如果没下载refcoco+、grefcoco等数据集的train文件则会报错，可以忽略。


### 获取预训练权重
首先需要用户在mmdetection目录下创建weights文件夹。

1. 用户可访问huggingface官网自行下载bert模型，在./weights目录下创建bert文件夹

   将bert模型文件放置在./weights/bert目录下
   
2. NLTK权重

   参考[链接](https://github.com/nltk/nltk_data)

   将packages改名为nltk_data放在~/路径下

3. MM-GDINO-B权重
   
   下载参考[链接](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino) 
   
   请下载表“Zero-Shot COCO Results and Models”中MM-GDINO-B的Zero-shot模型，模型文件名为：grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
   
   将MM-GDINO-B权重放置在./weights/目录下

## 快速开始

### 微调任务

主要提供基于refcoco数据集全量微调的8卡训练脚本。

#### 开始训练

1. 修改groundingdino_npu/mm_grounding_dino_swin-b_finetune_b2_refcoco.py脚本中的数据集路径和权重路径配置，以下为实际使用示例：

   ```
   lang_model_name = './weights/bert'
   data_root = './refcoco/'
   load_from = "./weights/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"
   ```
   
2. 修改groundingdino_npu/finetune_refcoco.sh

   ```
   # 修改 ascend-toolkit 路径
   source groundingdino_npu/env_npu.sh 
   
   # 修改 Python 路径
   PYTHON_PATH="Python Env Path"
   ```

3. 启动预训练脚本
   ```
   bash groundingdino_npu/finetune_refcoco.sh
   ```

#### 训练结果

基于RefCOCO数据集训练2000步验证下游任务，具体结果见表3和表4：

**表 3**  精度结果展示表

|    芯片    |   AP   | Precision @ 1 | Precision @ 5 | Precision @ 10 | 
|:--------:|:------:|:-------------:|:-------------:|:--------------:|
|   GPU    | 0.3148 |    0.8554     |    0.9839     |     0.9938     |
| Atlas A2 | 0.3152 |    0.8588     |    0.9836     |     0.9941     | 


**表 4**  性能结果展示表

|    芯片    | 卡数 |  FPS  | batch_size | AMP_Type | Torch_Version |
|:--------:|:--:|:-----:|:----------:|:--------:| :-----------: |
|   GPU    | 8p | 13.13 |     2      |   fp32   |      2.1      |
| Atlas A2 | 8p | 8.45  |     2      |   fp32   |      2.1      |



### 推理任务

#### 开始推理

1. 修改推理配置文件中 groundingdino_npu/mm_grounding_dino_swin-b_inference.py 的模型路径

   ```
   lang_model_name = './weights/bert'
   ```
   
2. 修改 groundingdino_npu/inference.sh 的脚本权重路径，Python环境路径

   ```
   # 修改 ascend-toolkit 路径
   source groundingdino_npu/env_npu.sh 
   
   # 修改 Python 路径
   PYTHON_PATH="Python Env Path"
   
   export Mx_Driving_PYTHON_PATH=${PYTHON_PATH}/lib/python3.8
   export ASCEND_CUSTOM_OPP_PATH=${Mx_Driving_PYTHON_PATH}/site-packages/mx_driving/packages/vendors/customize
   export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:$LD_LIBRARY_PATH
   python groundingdino_npu/image_demo_npu.py \
          demo/demo.jpg \
          groundingdino_npu/mm_grounding_dino_swin-b_inference.py \
          # 修改 权重路径
          --weights ./weights/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth \
          --texts 'bench . car .'
   ```

3. 启动推理脚本
   ```
   bash groundingdino_npu/inference.sh
   ```

# 公网地址说明
暂无。

# 变更说明
暂无。


# FAQ

暂无。
