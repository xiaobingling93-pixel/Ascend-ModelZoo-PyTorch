# SAM2模型离线推理指导

## 概述
SAM2是Meta开源的多模态视觉分割模型，对比SAM1不仅优化了图像分割的精度，还新增了视频实时分割能力。
本文档以 sam2_hiera_tiny 的图像分割功能为例，介绍离线推理步骤。
- 参考实现：

  ```
  url=https://github.com/facebookresearch/sam2.git
  commit_id=2b90b9f5ceec907a1c18123530e92e794ad901a4
  model_name=sam2_hiera_tiny
  ```

## 推理环境准备

- 该模型需要以下插件与驱动。

**表 1**  版本配套表

  | 配套 | 版本 | 环境准备指导 |
  | ---- | ---- | ---- |
  | 固件与驱动 | 25.3.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 8.3.RC2 | - |
  | Python | 3.11.10 | - |
  | PyTorch | 2.5.1 | - |
  | Ascend Extension PyTorch | 2.5.1.post1 | - |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \ | \ |

## 快速上手

### 1. 获取源码
   
```bash
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git reset --hard 2b90b9f5ceec907a1c18123530e92e794ad901a4
mv ../sam2_coco_metric.py .
mv ../sam2_image_predictor_export_onnx.py .
mv ../sam2_postprocessing.py .
mv ../sam2_preprocessing.py .
```

### 2. 安装依赖

- 安装基础环境。

```bash
pip3 install -r ../requirements.txt
```
说明：如果某些库通过此方式安装失败，可使用 pip3 install 单独进行安装。

### 3. 输入输出数据描述

SAM2 支持图像分割功能，首先会自动分割图像中的所有内容，但是如果你需要分割某一个目标物体，则需要你输入一个目标物体上的坐标，比如一张图片你想让SAM分割Cat或Dog这个目标的提示坐标，SAM2会自动在照片中猫或狗进行分割，在离线推理时，会转成encoder模型和decoder模型，其输入输出详情如下：

- encoder输入数据

  | 输入数据 | 数据类型     | 大小               | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | image    | FLOAT32 | bs x 3 x 1024 x 1024 | NCHW         |

- encoder输出数据

  | 输出数据 | 数据类型         | 大小 | 数据排布格式 |
  |-----------| -------- | -------- | ------------ |
  | high_res_feats_0  | FLOAT32  |  bs x 32 x 256 x 256 | NCHW           |
  | high_res_feats_1  | FLOAT32  |  bs x 64 x 128 x 128 | NCHW           |
  | image_embed  | FLOAT32  |  bs x 256 x 64 x 64  | NCHW           |


- decoder输入数据

  | 输入数据 | 数据类型     | 大小               | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | high_res_feats_0  | FLOAT32  |  1 x 32 x 256 x 256 | NCHW           |
  | high_res_feats_1  | FLOAT32  |  1 x 64 x 128 x 128 | NCHW           |
  | image_embed  | FLOAT32  |  1 x 256 x 64 x 64  | NCHW           |
  | point_coords    | FLOAT32 | 1 x pointnums x 2 | ND         |
  | point_labels    | int8 | 1 x pointnums | ND         |
  | mask_input    | FLOAT32 | 1 x 1 x 256 x 256 | NCHW         |
  | has_mask_input    | int8 | 1 | ND         |


- decoder输出数据

  | 输出数据 | 数据类型        | 大小 | 数据排布格式 |
  |-----------| -------- | -------- | ------------ |
  | masks  | FLOAT32  | 1, 1, 256, 256  | ND           |
  | iou_predictions  | FLOAT32  |  1, 1 | ND           |
  | low_res_masks  | FLOAT32  | 1, 1, 256, 256  | ND           |


### 4. 准备数据集

```bash
mkdir coco2017
```

下载并解压COCO-2017数据集的[图片](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fzips%2Fval2017.zip)与[标注](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fannotations%2Fannotations_trainval2017.zip)，放置coco2017目录下。
   
   ```
   coco2017
   ├── annotations/
   │   └── instances_val2017.json
   └── val2017/
       ├── 000000000139.jpg
       ├── 000000000139.jpg
       └── ...
   ```

### 5. 模型转换

#### 5.1 获取权重文件

选择下载[模型权重](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints)，下面步骤以sam2.1_hiera_tiny.pt为例，下载权重文件[sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)，放到checkpoints目录下。


#### 5.2 导出 ONNX 模型

```bash
mkdir models
mkdir models/onnx

bs=1
# encoder模型的batch_size，默认取1
pointnums=2
# decoder模型输入的提示点数量，默认取2

python sam2_image_predictor_export_onnx.py \
  --model_cfg=configs/sam2/sam2_hiera_t \
  --checkpoint=checkpoints/sam2_hiera_tiny.pt \
  --opset=14 \
  --encoder_output=models/onnx/sam2_hiera_tiny_encoder_bs${bs}.onnx \
  --decoder_output=models/onnx/sam2_hiera_tiny_decoder_pointnums${pointnums}.onnx \
  --bs=${bs} \
  --pointnums=${pointnums}
```

参数说明：

- checkpoint：模型权重文件路径。
- model_cfg：模型配置文件。
- opset：onnx算子集版本。
- encoder_output：保存encoder模型的输出ONNX模型的文件路径。
- decoder_output：保存decoder模型的输出ONNX模型的文件路径。
- bs: encoder模型的batch_size，默认1。
- point_nums: decoder模型输入的提示点数量，默认2。

#### 5.3 使用 onnxsim 简化 ONNX 模型

```bash
onnxsim models/onnx/sam2_hiera_tiny_encoder_bs${bs}.onnx models/onnx/sam2_hiera_tiny_encoder_bs${bs}_sim.onnx
onnxsim models/onnx/sam2_hiera_tiny_decoder_pointnums${pointnums}.onnx models/onnx/sam2_hiera_tiny_decoder_pointnums${pointnums}_sim.onnx
```

参数说明：

- 第一个参数：原 ONNX 模型路径。
- 第二个参数：简化后的 ONNX 模型保存路径。


#### 5.4 使用 ATC 工具将 ONNX 模型转为 OM 模型

- 配置环境变量。

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

- 执行命令查看芯片名称（$\{chip\_name\}）。

   ```bash
   npu-smi info
   #该设备芯片名为Ascend310P3 （自行替换）
   chip_name=310P3
   回显如下：
   +-------------------+-----------------+------------------------------------------------------+
   | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
   | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
   +===================+=================+======================================================+
   | 0       310P3     | OK              | 15.8         42                0    / 0              |
   | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
   +===================+=================+======================================================+
   | 1       310P3     | OK              | 15.4         43                0    / 0              |
   | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
   +===================+=================+======================================================+
   ```

- 执行 atc 命令。

   ```bash
   mkdir models/om
   
   atc \
   --framework 5 \
   --model models/onnx/sam2_hiera_tiny_encoder_bs${bs}_sim.onnx \
   --output models/om/sam2_hiera_tiny_encoder_bs${bs}_sim \
   --input_shape "image:${bs},3,1024,1024" \
   --enable_small_channel 1 \
   --soc_version "Ascend${chip_name}"

   atc \
    --framework 5 \
    --model models/onnx/sam2_hiera_tiny_decoder_pointnums${pointnums}_sim.onnx \
    --output models/om/sam2_hiera_tiny_decoder_pointnums${pointnums}_sim \
    --input_shape "image_embed:1,256,64,64;high_res_feats_0:1,32,256,256;high_res_feats_1:1,64,128,128;point_coords:1,${pointnums},2;point_labels:1,${pointnums};mask_input:1,1,256,256;has_mask_input:1" \
    --soc_version "Ascend${chip_name}"
   ```

   参数说明：

   - framework：原始框架类型。
   - model：原始模型文件路径与文件名。
   - output：存放转换后的离线模型的路径以及文件名。
   - enable_small_channel：是否使能small channel的优化，使能后在channel<=4的首层卷积会有性能收益。
   - input_shape：模型输入数据的shape。
   - soc_version：模型转换时指定芯片版本。

### 6. 推理验证

#### 6.1 安装ais_bench推理工具

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

#### 6.2 性能验证

- encoder 纯推理性能验证。

    ```bash
    python3 -m ais_bench --model models/om/sam2_hiera_tiny_encoder_bs${bs}_sim.om --loop 100
    ```

    参数说明：

    - model: OM 模型路径。
    - loop: 循环次数。

- decoder 纯推理性能验证。

    ```bash
    python3 -m ais_bench --model models/om/sam2_hiera_tiny_decoder_pointnums${pointnums}_sim.om --loop 100
    ```

    参数说明：

    - model: OM 模型路径。
    - loop: 循环次数。

#### 6.3 精度验证
   ```bash
    python sam2_coco_metric.py \
        --dataset_path=coco2017 \
        --save_path=outputs \
        --encoder_model_path=models/om/sam2_hiera_tiny_encoder_bs${bs}_sim.om \
        --decoder_model_path=models/om/sam2_hiera_tiny_decoder_pointnums${pointnums}_sim.om \
        --device-id=0 \
        --bs=${bs} \
        --max_instances=0
   ```

参数说明：

- dataset_path: coco数据集目录。
- save_path: 预测掩码结果存储路径。
- encoder_model_path：encoder的OM模型路径。
- decoder_model_path：decoder的OM模型路径。
- device_id: 指定推理的NPU设备ID。
- bs: encoder模型的batch size。
- max_instances: 评测的最大实例数量，默认为0表示测评完整验证集。

## 模型推理性能 & 精度
encoder模型性能结果：
| 芯片型号 | 模型 | bs | 性能(fps) |
| ---- | ---- | ---- | ---- |
| 300I DUO | sam2_hiera_tiny | 1 | 9.83 |
| 300I DUO | sam2_hiera_tiny | 4 | 10.81 |
| 300I DUO | sam2_hiera_tiny | 8 | 9.19 |

decoder模型性能结果：
| 芯片型号 | 模型 | bs | 性能(fps) |
| ---- | ---- | ---- | ---- |
| 300I DUO | sam2_hiera_tiny | 1 | 476.97 |

精度结果：
| 芯片型号 | 模型 | 数据集 | bs(encoder 模型) | 精度（mIoU） | GPU精度（mIoU） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 300I DUO | sam2_hiera_tiny | coco2017 | 1 | 0.7828 | 0.7830 |
| 300I DUO | sam2_hiera_tiny | coco2017 | 4 | 0.7828 | 0.7830 |
| 300I DUO | sam2_hiera_tiny | coco2017 | 8 | 0.7828 | 0.7830 |