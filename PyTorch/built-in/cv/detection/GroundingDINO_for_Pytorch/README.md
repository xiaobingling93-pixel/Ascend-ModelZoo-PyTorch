# Grounding DINO for Pytorch

# 目录

-   [简介](#简介)
    -   [模型介绍](#模型介绍)
    -   [支持任务列表](#支持任务列表)
    -   [代码实现](#代码实现)
-   [Grounding-dino](#Grounding-dino)
    -   准备训练环境
    -   准备数据集
    -   快速开始
        -   微调任务
        -   推理任务
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)


# 简介

## 模型介绍

Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

Grounding DINO模型可以根据输入文本检测文本对应的目标.

官方仓：https://github.com/IDEA-Research/GroundingDINO

## 支持任务列表

本仓已支持以下模型任务类型。

| 模型             | 任务类型       | 是否支持  |
|----------------|------------| ------------ |
| Grounding-dino | 微调         | ✅   |



## 代码实现

- 参考实现：

  ```
  MMDetection仓：https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection/GroundingDINO_for_Pytorch
  ```

# Grounding-dino
## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|        软件类型        |   支持版本   |
|:------------------:|:--------:|
| FrameworkPTAdapter | 6.0.RC1  |
|        CANN        | 8.0.RC1  |
|      昇腾NPU固件       | 24.1.RC1 |
|      昇腾NPU驱动       | 24.1.RC1 |
|        ADS         |  v2.1.0  |

### 安装模型环境

**表 2**  三方库版本支持表

|    三方库    |  支持版本  |
|:---------:|:------:|
|  PyTorch  | 2.1.0  |


在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。
需要先安装PTA和ADS包。

```shell
pip install torchvision==0.16.0
pip install opencv-python
pip install opencv-python-headless
git clone https://github.com/open-mmlab/mmcv mmcv 
cd mmcv
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r requirements.txt
MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext 
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
cd ..
pip install mmengine==0.10.3
git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 mmdetection
cd mmdetection
pip install -r requirements.txt
```

将groundingdino_npu目录放在mmdetection目录下。
```shell
pip install -r groundingdino_npu/requirements.txt
```


### 准备数据集

#### 微调数据集
训练与评估所使用的数据集为refcoco数据集，
用户需自行获取COCO2014数据集，将训练集命名为train2014, refcoco对应的标注数据下载[下载地址](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations)，
创建文件夹mdetr_annotaions并将这四个json文件放入其中。

准备好后文件夹结构如下所示：

```text

├refcoco
├── annotations
│   ├── instances_train2014.json
│   ├── ...
├── train2014
│   ├── xxx.jpg
│   ├── ...
├── val2014
│   ├── xxxx.jpg
│   ├── ...
├── mdetr_annotations
│   ├── finetune_refcoco_val.json
│   ├── finetune_refcoco_testA.json
│   ├── finetune_refcoco_testB.json
│   ├── finetune_refcoco_train.json
```

然后需要使用以下命令将train.json文件转换为所需的 ODVG 格式：
```shell
cd $mmdetection
python tools/dataset_converters/refcoco2odvg.py refcoco/mdetr_annotations
```
以上命令会在mdetr_annotations文件夹下生成finetune_refcoco_train_vg.json文件，如果没下载refcoco+、grefcoco等数据集的train文件则会报错，可以忽略。


### 获取预训练权重

1. 用户可访问huggingface官网自行下载bert模型，文件namespace如下：
   ```
   bert-base-uncased
   ```
2. NLTK权重

   参考[链接](https://github.com/nltk/nltk_data)

   将packages改名为nltk_data放在~/路径下

3. MM-GDINO-B权重
   
   下载参考[链接](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino)
   表“Zero-Shot COCO Results and Models”

## 快速开始

### 微调任务

主要提供基于refcoco数据集全量微调的8卡训练脚本。

#### 开始训练

修改groundingdino_npu/mm_grounding_dino_swin-b_finetune_b2_refcoco.py脚本中的数据集路径和权重路径配置.

```
PYTHON_PATH="Python Env Path"
export ADS_PYTHON_PATH=${PYTHON_PATH}/lib/python3.8
export ASCEND_CUSTOM_OPP_PATH=${ADS_PYTHON_PATH}/site-packages/ads/packages/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:$LD_LIBRARY_PATH

lang_model_name = '../bert-base-uncased/'
data_root = '../refcoco/'
load_from = "../weights/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"
```

修改groundingdino_npu/finetune_refcoco.sh并执行.


#### 训练结果

##### 精度

基于RefCOCO数据集训练2000步验证下游任务：

|    芯片    |   AP   | Precision @ 1 | Precision @ 5 | Precision @ 10 | 
|:--------:|:------:|:-------------:|:-------------:|:--------------:|
|   GPU    | 0.3148 |    0.8554     |    0.9839     |     0.9938     |
| Atlas A2 | 0.3152 |    0.8588     |    0.9836     |     0.9941     | 

##### 性能


|    芯片    | 卡数 |  FPS  | batch_size | AMP_Type | Torch_Version |
|:--------:| :----: |:-----:|:----------:|:--------:| :-----------: |
|   GPU    |   8p   | 13.13 |     2      |   fp32   |      2.1      |
| Atlas A2 |   8p   | 8.45  |     2      |   fp32   |      2.1      |



### 推理任务

#### 开始推理

修改推理配置文件中groundingdino_npu/mm_grounding_dino_swin-b_inference.py的模型路径
```
lang_model_name = '../bert-base-uncased/'
```


修改inference.sh脚本权重路径，Python环境路径并执行

    source groundingdino_npu/env_npu.sh
    PYTHON_PATH="Python Env Path"
    export ADS_PYTHON_PATH=${PYTHON_PATH}/lib/python3.8
    export ASCEND_CUSTOM_OPP_PATH=${ADS_PYTHON_PATH}/site-packages/ads/packages/vendors/customize
    export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:$LD_LIBRARY_PATH
    
    python groundingdino_npu/image_demo_npu.py \
        demo/demo.jpg \
        groundingdino_npu/mm_grounding_dino_swin-b_inference.py \
        --weights 权重路径 \
        --texts 'bench . car .'


# 公网地址说明
暂无。

# 变更说明
暂无。


# FAQ

暂无。

