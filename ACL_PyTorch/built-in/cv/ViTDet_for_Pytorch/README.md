# ViTDet(TorchAir)-推理指导

- [概述](#summary)
  
  - [输入数据](#input_data)

- [推理环境准备](#env_setup)

- [快速上手](#quick_start)
  
  - [获取源码](#get_code)
  
  - [下载数据集](#download_data)
  
  - [模型推理](#infer)

- [模型推理性能 & 精度](#performance)

# 概述<a id="summary"></a>

ViTDet是Meta AI团队（kaiming团队）在MAE之后提出的基于原生ViT模型作为骨干网络的检测模型。在最早的论文[Benchmarking Detection Transfer Learning with Vision Transformers](https://arxiv.org/abs/2111.11429)中，作者初步研究了以ViT作为骨干网络的检测模型所面临的挑战（架构的不兼容，训练速度慢以及显存占用大等问题），并给出了具体的解决方案，最重要的是发现基于MAE的预训练模型展现了较强的下游任务迁移能力，效果大大超过随机初始化和有监督预训练模型。而最新的论文[Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527)对上述工作做了进一步的拓展和优化，给出了性能更好的ViTDet。本篇对第二篇论文提出的ViTDet模型的[具体实现](https://github.com/open-mmlab/mmdetection/tree/main/projects/ViTDet)进行Torch-Air相关的适配。

- 版本说明:
  
  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=cfd5d3a
  model_name=ViTDet
  ```

## 输入数据<a id="input_data"></a>

ViTDet使用公共数据集COCO进行推理

| 输入数据 | 数据类型     | 大小          | 数据排布格式 |
|:----:|:--------:|:-----------:|:------:|
| img  | RGB_FP32 | （1，3，-1，-1） | NCHW   |

# 

# 推理环境准备<a id="env_setup"></a>

该模型需要以下依赖

表1 **版本配套表**

| 依赖      | 版本            | 环境准备指导                                                                                                        |
| ------- | ------------- |:-------------------------------------------------------------------------------------------------------------:|
| 固件与驱动   | 25.0.rc1.b010 | [PyTorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies/pies_00001.html) |
| PyTorch | 2.4.0         | -                                                                                                             |
| CANN    | 8.1.RC1       | -                                                                                                             |
| Python  | 3.9           | -                                                                                                             |

# 快速上手<a id="quick_start"></a>

## 获取源码<a id="get_code"></a>

1. 获取本仓源码
   
   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/ViTDet_for_Pytorch
   ```

2. 获取模型仓**mmdetection**源码和依赖仓**mmengine**源码，并安装相关依赖
   
   ```
   git clone https://github.com/open-mmlab/mmdetection.git
   git clone https://github.com/open-mmlab/mmengine.git
   
   
   
   cd mmdetection
   git reset --hard cfd5d3a985b0249de009b67d04f37263e11cdf3d
   pip3 install -r requirements.txt
   
   cd ../mmengine
   git reset --hard 390ba2fbb272816adfd2883642326d0fd0ca6049
   pip3 install -r requirements.txt
   cd ..
   ```

3. 转移文件位置
   
   ```
   cp mmengine.patch mmengine/mmengine/
   cp mmdet.patch mmdetection/
   cp infer.py mmdetection/
   ```

4. 更换当前路径并打补丁，修改完mmseg源码后进行安装
   
   ```
   cd mmengine/mmengine/
   patch -p2 < mmengine.patch
   pip install -v -e ..
   
   cd ../../mmdetection
   patch -p1 < mmdet.patch
   ```

## 下载数据集与权重<a id="download_data"></a>

1. 下载权重文件并放到mmdetection/ckpt下
   
   > [ckpt文件下载](https://download.openmmlab.com/mmdetection/v3.0/vitdet/vitdet_mask-rcnn_vit-b-mae_lsj-100e/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth)

2. 使用下面的链接下载数据集并解压放在mmdetection/data目录下
   
   > [COCO数据集下载](https://cocodataset.org/#download)
   
   确保data下的路径结构如下
   
   ```
   ├── data
   │   ├── coco
   │   │   ├── annotations
   │   │   ├── val2017
   ```

## 模型推理<a id="infer"></a>

1. 设置环境变量
   
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. 执行推理命令
   
   ```
   python infer.py --cfg projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py --ckpt "ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth" --warm_up_times 1 
   ```
   
   - 参数说明
     
     - --cfg：配置文件路径
     
     - --ckpg：权重文件路径
     
     - --warm_up_times：正式推理前的warmup次数，默认为2
     
     - --loop：推理的循环次数，默认为0

# 模型推理性能 & 精度<a id="performance"></a>

| 芯片型号    | 模型     | box mAP | seg mAP | 纯推理性能    | 端到端性能    |
| ------- | ------ | ------- | ------- | -------- | -------- |
| 800I A2 | ViTDet | 51.3%   | 45.5%   | 113.68ms | 413.20ms |
