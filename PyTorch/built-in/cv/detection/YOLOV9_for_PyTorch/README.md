# YoloV9 for PyTorch

-   [简介](#简介)
    -   [模型介绍](#模型介绍)
    -   [支持任务列表](#支持任务列表)
    -   [代码实现](#代码实现)
-   [训练](#训练)
    -   [准备环境](#准备环境)
    -   [准备数据集](#准备数据集)
    -   [开始训练](#开始训练)
    -   [训练结果展示](#训练结果展示)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)


# 简介

## 模型介绍

YOLOv9融合了深度学习技术和架构设计的进步，以在对象检测任务中实现卓越的性能。将可编程梯度信息 (PGI) 概念与通用 ELAN (GELAN)架构相结合而开发，代表了准确性、速度和效率方面的重大飞跃。

主要特点：

1. 实时对象检测：YOLOv9 通过提供实时对象检测功能保持了 YOLO 系列的标志性功能。这意味着它可以快速处理输入图像或视频流，并准确检测其中的对象，而不会影响速度。
2. PGI集成：YOLOv9融合了可编程梯度信息（PGI）概念，有助于通过辅助可逆分支生成可靠的梯度。这确保深度特征保留执行目标任务所需的关键特征，解决深度神经网络前馈过程中信息丢失的问题。
3. GELAN架构：YOLOv9采用通用ELAN（GELAN）架构，旨在优化参数、计算复杂度、准确性和推理速度。通过允许用户为不同的推理设备选择合适的计算模块，GELAN 增强了 YOLOv9 的灵活性和效率。
4. 性能提升：实验结果表明，YOLOv9 在 MS COCO 等基准数据集上的目标检测任务中实现了最佳性能。它在准确性、速度和整体性能方面超越了现有的实时物体检测器，使其成为需要物体检测功能的各种应用的最先进的解决方案。
5. 灵活性和适应性：YOLOv9 旨在适应不同的场景和用例。其架构可以轻松集成到各种系统和环境中，使其适用于广泛的应用，包括监控、自动驾驶车辆、机器人等。

## 支持任务列表

本仓已支持以下模型任务类型。

| 模型  | 任务类型  | 是否支持  |
| ------------ | ------------ | ------------ |
| YOLOV9  | 训练  | ✅   |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/WongKinYiu/yolov9.git
  commit_id=5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection/YOLOV9_for_PyTorch/
  ```

# 训练

## 准备环境

 - 安装昇腾环境。

   请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

 - 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

   **表 1**  版本支持表

   | Torch_Version      | 三方库依赖版本                  |
   | :--------: | :----------------------------------------------------------: |
   | PyTorch 1.11 | torchvision==0.12.0；torchvision_npu==0.12.0 |
   | PyTorch 2.1 | torchvision==0.16.0；torchvision_npu==0.16.0 |

   **表 2**  昇腾软件版本支持表

   |      软件类型      |                           支持版本                           |
   | :----------------: | :----------------------------------------------------------: |
   | FrameworkPTAdapter | 6.0.RC1/在研版本 |
   |        CANN        | 8.0.RC1/在研版本 |
   |    昇腾NPU固件      | 24.1.RC1/在研版本 |
   |    昇腾NPU驱动      | 24.1.RC1/在研版本 |

 - 安装依赖。

   在YOLOV9_for_PyTorch目录下执行命令，安装模型对应PyTorch版本需要的依赖。
   ```
   pip install torchvision==0.12.0 # torch 1.11
   # pip install torchvision==0.16.0 # torch 2.1
   cd PyTorch/built-in/cv/detection/YOLOV9_for_PyTorch
   pip install -r requirements.txt
   ```
   > **说明：** 
   >只需执行一条对应的PyTorch版本依赖安装命令。

 - 安装torchvision_npu。

      ```
      cd ..
      git clone -b v0.12.0-dev https://gitee.com/ascend/vision.git torchvision_npu # torch 1.11
      # git clone https://gitee.com/ascend/vision.git torchvision_npu # torch 2.1

      cd torchvision_npu
      pip install -r requirement.txt
      python setup.py bdist_wheel
      pip install dist/torchvision_npu*.whl
      ```

## 准备数据集

1. 获取数据集。

    a. 下载 [Arial.ttf](https://ultralytics.com/assets/Arial.ttf) 文件上传至服务器 /root/.config/Ultralytics/Arial.ttf。

    b. 联网情况下，在YOLOV9_for_PyTorch目录执行bash scripts/get_coco.sh，自动下载coco2017数据集，然后移动coco至YOLOV9_for_PyTorch同级别的datasets目录下，数据集目录结构参考如下所示。

    c. 无网络情况下，下载[coco2017labels-segments.zip](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip)并解压至datasets目录，下载[val2017.zip](http://images.cocodataset.org/zips/val2017.zip)、[test2017.zip](http://images.cocodataset.org/zips/test2017.zip)、[train2017.zip](http://images.cocodataset.org/zips/train2017.zip)并解压至datasets/coco/images目录。

  ```shell script
  ├── datasets #数据集目录
    ├── coco #coco数据集
      ├── train2017.txt #训练集图片列表
      ├── val2017.txt #验证集图片列表
      ├── images #图片
        ├── train2017 #训练集图片，118287张
        ├── val2017 #验证集图片，5000张
        ├── test2017 #测试集图片，40670张
  ├── YOLOV9_for_PyTorch #根目录
    ├── scripts
      ├── get_coco.sh #下载数据集脚本
   ```

## 开始训练

1. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     ```
     export CPU_AFFINITY_CONF=1 # CPU绑核，可选
     python -m torch.distributed.launch --nproc_per_node 1 --master_port 9527 train_dual.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 100 --close-mosaic 15 --noplots
     ```

   - 单机8卡训练

     ```
     export CPU_AFFINITY_CONF=1 # CPU绑核，可选
     python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --batch 256 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 100 --close-mosaic 15 --noplots
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --nproc_per_node                 //使用的npu卡数量
   --device                         //训练设备卡号
   --batch                          //单步训练图片数量
   --epochs                         //训练次数
   --data                           //数据集配置
   --cfg                            //训练配置
   --name                           //训练名
   ```
   训练完成后，权重文件保存在runs/train，并输出模型训练精度和性能信息。

## 训练结果展示

**表 3**  训练性能

| NAME     | 卡数 | Train Times(1 epoch) | Train Times(1 epoch) | Torch_Version |
|:--------:| :---: | :---: | :---: | :---: |
| 竞品V          | 8p | 05:12 | 00:33 | 2.4 |
| Atlas 800T A2 | 8p | 07:23 | 01:44 | 1.11 |

**表 4**  训练精度
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.702
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.578
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.652
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.702
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.541
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.760
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.844
```


# 公网地址说明

代码涉及公网地址参考 [public_address_statement.md](public_address_statement.md)

# 变更说明

2024.8.13：首次发布。

# FAQ
