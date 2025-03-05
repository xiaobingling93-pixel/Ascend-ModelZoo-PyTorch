# YOLOV11(TorchAir)-推理指导

- [YOLOV11(TorchAir)-推理指导](#yolov11torchair-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [获取数据集](#获取数据集)
  - [获取权重](#获取权重)
  - [执行推理](#执行推理)
  - [性能数据](#性能数据)

******

# 概述

- 版本说明：
  ```
  url=https://github.com/ultralytics/ultralytics.git
  commit_id=e74b035
  model_name=YOLOV11
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 24.0.RC3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  8.1.0 | -                                                                                                   |
  | Python                                                          |  3.8 | -                                                                                                     |
  | PyTorch                                                         | 2.1.0 | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.1.0.post8 | -                                                                                                     |
  | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git reset --hard e74b035
git apply ../diff.patch
cd ..
export PYTHONPATH=./ultralytics:$PYTHONPATH
```
   
2. 安装依赖  
```
pip3 install -r requirements.txt
```


## 获取数据集

1. 新建datasets文件夹，下载COCO-2017数据集，放置datasets目录下
```
 ultralytics
  └── dataset
      └── coco-2017
```

3. 利用dataset.py脚本，将数据集转化为YOLO仓支持的格式
```bash
python3 dataset.py --data_path=./coco/annotations/
```
- 参数说明
  - data_path: coco数据集annotations目录

执行后生成coco_converted文件夹

3. 移动image文件夹到coco_converted内
```
mv coco/val2017 coco_converted/image
```
完成后数据集结构如下
```
 dataset
 └── coco_converted
      ├── image
            └── val2017
                └── ***.jpg
      └── label
          └── val2017
                  └── ***.txt
```

## 获取权重
下载权重文件[yolo11l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt)，放到当前目录下
```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
```

## 执行推理
运行推理脚本infer.py
```
python3 infer.py --pth=yolo11l.pt --dataset=coco.yaml --batchsize=16
```
- 参数说明
  - pth: 模型权重，以yolo11l.pt为例
  - dataset: 数据集信息
  - batchsize: 数据集推理batchsize
  
推理执行完成后，数据集精度和数据集性能会执行打屏

## 性能数据
以bs16在数据集上的推理为例
|模型|芯片|性能(per image)|精度(mAP)|
|------|------|------|------|
|yolo11l|800I A2|1.7ms|53.3|

注：该模型支持动态shape数据集推理，输入shape长边会处理为640，短边以原尺寸scale


