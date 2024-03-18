# SAM 推理指导

Segment Anything Model (SAM) 是由 Meta 开源的图像分割大模型，在计算机视觉领域（CV）取得了新的突破。SAM 可在不需要任何标注的情况下，对任何图像中的任何物体进行分割，SAM 的开源引起了业界的广泛反响，被称为计算机视觉领域的 GPT。

- 论文：

  ```
  [Segment Anything](https://arxiv.org/abs/2304.02643)
  Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
  ```

- 参考实现：

  ```
  url=https://github.com/facebookresearch/segment-anything.git
  commit_id=6fdee8f2727f4506cfbbe553e23b895e27956588
  model_name=sam_vit_b_01ec64
  ```

## 1. 输入输出数据

SAM 首先会自动分割图像中的所有内容，但是如果你需要分割某一个目标物体，则需要你输入一个目标物体上的坐标，比如一张图片你想让SAM分割Cat或Dog这个目标的提示坐标，SAM会自动在照片中猫或狗进行分割，在离线推理时，会转成encoder模型和decoder模型，其输入输出详情如下：

- encoder输入数据

  | 输入数据 | 数据类型     | 大小               | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | x    | FLOAT32 | 1 x 3 x 1024 x 1024 | NCHW         |

- encoder输出数据

  | 输出数据 | 数据类型         | 大小 | 数据排布格式 |
  |-----------| -------- | -------- | ------------ |
  | image_embeddings  | FLOAT32  |  1 x 256 x 64 x 64 | NCHW           |


- decoder输入数据

  | 输入数据 | 数据类型     | 大小               | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | image_embeddings    | FLOAT32 | 1 x 256 x 64 x 64 | NCHW         |
  | point_coords    | FLOAT32 | 1 x -1 x 2 | ND         |
  | point_labels    | FLOAT32 | 1 x -1 | ND         |
  | mask_input    | FLOAT32 | 1 x 1 x 256 x 256 | NCHW         |
  | has_mask_input    | FLOAT32 | 1 | ND         |


- decoder输出数据

  | 输出数据 | 数据类型        | 大小 | 数据排布格式 |
  |-----------| -------- | -------- | ------------ |
  | iou_predictions  | FLOAT32  | -1 x 1  | ND           |
  | low_res_masks  | FLOAT32  |  -1 x 1 x -1 x -1 | ND           |


## 2. 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套 | 版本 | 环境准备指导 |
| ---- | ---- | ---- |
| 固件与驱动 | 24.0.T1.B010 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN | 8.0.RC1.B030 | - |
| MindIE | 1.0.RC1.B050 | - |
| Python | 3.10.13（MindIE 要求 Python 3.10） | - |
| PyTorch | 1.13.1 | - |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \ | \ |

## 3. 快速上手

### 3.1 获取源码

```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
git reset --hard 6fdee8f2727f4506cfbbe553e23b895e27956588
pip3 install -e .
cd ..
patch -p1 < segment_anything_diff.patch
```

### 3.2 安装依赖。

1. 安装基础环境。

   ```bash
   pip3 install -r requirements.txt
   ```

   说明：如果某些库通过此方式安装失败，可使用 pip3 install 单独进行安装。

2. 安装 [ait](https://gitee.com/ascend/ait/tree/master/ait) 的 surgeon 组件和 benchmark 组件。

   ```bash
   git clone https://gitee.com/ascend/ait.git
   cd ait/ait
   chmod u+x install.sh
   ./install.sh --surgeon
   ./install.sh --benchmark
   cd ../..
   ```

3. 安装量化工具。

   参考[AMCT(ONNX)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha002/developmenttools/devtool/atlasamctonnx_16_0001.html)主页安装量化工具。

   说明：AMCT工具和CANN包版本配套，建议onnxruntime用1.8.0版本。

### 3.3 准备数据集

GitHub 仓库没有提供精度和性能的测试手段，这里取仓库里的 demo 图片进行测试。

```bash
mkdir data
cd data
wget -O demo.jpg https://raw.githubusercontent.com/facebookresearch/segment-anything/571794162e0887c15d12b809505b902c7bf8b4db/notebooks/images/truck.jpg
cd ..
```

### 3.4 模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

#### 3.4.1 获取权重文件

GitHub 仓库提供了三种大小的权重文件：vit_h、vit_l、vit_b。这里以 vit_b 为例。

```bash
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..
```

#### 3.4.2 导出 ONNX 模型

```bash
python3 segment-anything/scripts/export_onnx_model.py  \
  --checkpoint models/sam_vit_b_01ec64.pth  \
  --model-type vit_b  \
  --opset 14 \
  --encoder-output models/encoder.onnx \
  --decoder-output models/decoder.onnx \
  --return-single-mask
```

参数说明：

- checkpoint：模型权重文件路径。
- model-type：模型类型。
- opset：onnx算子集版本。
- encoder-output：保存encoder模型的输出ONNX模型的文件路径。
- decoder-output：保存decoder模型的输出ONNX模型的文件路径。
- return-single-mask：设置最优mask模式。

#### 3.4.3 使用 onnxsim 简化 ONNX 模型

这里以 batchsize=1 为例。

```bash
batchsize=1
onnxsim models/encoder.onnx models/encoder_sim.onnx --overwrite-input-shape="x:${batchsize},3,1024,1024"
onnxsim models/decoder.onnx models/decoder_sim.onnx
```

参数说明：

- 第一个参数：原 ONNX 路径。
- 第二个参数：简化后的 ONNX 保存路径。
- overwrite-input-shape：指定输入的维度。

#### 3.4.4 运行改图脚本，修改 ONNX 模型以适配昇腾芯片

```bash
python3 encoder_onnx_modify.py \
  --input models/encoder_sim.onnx \
  --output models/encoder_modify.onnx
python3 decoder_onnx_modify.py \
  --input models/decoder_sim.onnx \
  --output models/decoder_modify.onnx
```

参数说明：

- 第一个参数：原 ONNX 路径。
- 第二个参数：适配后的 ONNX 保存路径。

#### 3.4.5 使用 ATC 工具将 ONNX 模型转为 OM 模型

1. 配置环境变量。

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/mindie-rt/set_env.sh
   ```

   > **说明：** 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

2. 执行命令查看芯片名称（$\{chip\_name\}）。

   ```bash
   npu-smi info
   #该设备芯片名为Ascend310P3 （自行替换）
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

3. 执行 atc 命令。

   ```bash
   atc \
     --framework 5 \
     --model models/encoder_modify.onnx \
     --output models/encoder_modify \
     --insert_op_conf sam.aippconfig \
     --enable_small_channel 1 \
     --soc_version "Ascend${chip_name}"
   atc \
     --framework 5 \
     --model models/decoder_modify.onnx \
     --output models/decoder_modify \
     --input_format 'ND' \
     --input_shape 'image_embeddings:1,256,64,64;point_coords:1,-1,2;point_labels:1,-1;mask_input:1,1,256,256;has_mask_input:1' \
     --dynamic_dims '2,2;3,3;4,4;5,5;6,6;7,7;8,8;9,9' \
     --soc_version "Ascend${chip_name}"
   ```

   参数说明：

   - framework：原始框架类型。
   - model：原始模型文件路径与文件名。
   - output：存放转换后的离线模型的路径以及文件名。
   - insert_op_conf：插入算子的配置文件路径与文件名，例如aipp预处理算子。
   - enable_small_channel：是否使能small channel的优化，使能后在channel<=4的首层卷积会有性能收益。
   - input_format：输入数据格式。
   - input_shape：模型输入数据的shape。
   - dynamic_dims：设置ND格式下动态维度的档位。适用于执行推理时，每次处理任意维度的场景。
   - soc_version：模型转换时指定芯片版本。

   更多参数说明请参考 [ATC 参数概览](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devaids/auxiliarydevtool/atlasatc_16_0039.html)（如果链接失效，请从 [CANN 社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition) 查找目录「应用开发 > ATC 模型转换 > 参数说明 > 参数概览」）。

#### 3.4.6 模型量化

<details>

  <summary>点击查看量化的详细步骤</summary>

1. encoder模型量化

   在量化前，先生成校验数据，以确保量化后模型精度不会损失：

   ```
   export PYTHONPATH="${PYTHONPATH}:`pwd`/segment-anything"
   python3 sam_quant_preprocessing.py \
         --src-path ./segment-anything/data/demo.jpg \
         --encoder-quant-save-path ./segment-anything/encoder_quant_bin \
         --encoder-quant
   
   ```
   - 参数说明
   - --src_path: 数据地址
   - --encoder-quant-save-path: 生成校验数据bin文件地址
   - --encoder-quant: 生成encoder量化的校验数据

   然后使用`amct`工具，对ONNX模型进行量化，以进一步提升模型性能：

   ```
   amct_onnx calibration \
         --model=./segment-anything/models/encoder_sim.onnx \
         --save_path=./segment-anything/models/encoder_quant \
         --input_shape="x:1,3,1024,1024" \
         --data_dir=./segment-anything/encoder_quant_bin/x \
         --data_types="float32"
   
   ```

   - 参数说明
   - --model: onnx模型
   - --save_path: 保存量化后onnx模型文件地址
   - --input_shape: 模型输入shape
   - --data_dir: 校验数据
   - --data_types: 数据类型
     

   量化后的模型存放路径为 `models/encoder_quant_deploy_model.onnx`。

2. decoder模型量化

   在量化前，先生成校验数据，以确保量化后模型精度不会损失：

   ```
   python3 sam_quant_preprocessing.py \
         --src-path ./segment-anything/data/demo.jpg \
         --encoder-onnx-model-path ./segment-anything/models/encoder_sim.onnx \
         --decoder-quant-save-path ./segment-anything/decoder_quant_bin \
         --input-point "[[500, 375], [1125, 625], [1520, 625]]" \
         --decoder-quant
   ```

   - 参数说明
   - --src_path: 测试数图片地址（自己根据实际图片路径进行更改）
   - --decoder-quant-save-path: 生成校验数据bin文件地址
   - --encoder-onnx-model-path: encoder模型路径
   - --input-point ：输入坐标，根据实际待分割目标物体上指定的点进行修改
   - --decoder-quant: 生成decoder量化的校验数据

   然后使用`amct`工具，对ONNX模型进行量化，以进一步提升模型性能：

   ```
   decoder_q=./segment-anything/decoder_quant_bin
   amct_onnx calibration \
         --model=./segment-anything/models/decoder_modify.onnx \
         --save_path=./segment-anything/models/decoder_quant \
         --input_shape="image_embeddings:1,256,64,64;point_coords:1,-1,2;point_labels:1,-1;mask_input:1,1,256,256;has_mask_input:1"  \
         --data_dir="${decoder_q}/image_embedding;${decoder_q}/point_coord;${decoder_q}/point_label;${decoder_q}/mask_input;${decoder_q}/has_mask_input" \
         --data_types="float32;float32;float32;float32;float32" 
   ```

   - 参数说明
      - --model: onnx模型
      - --save_path: 保存量化后onnx模型文件地址
      - --input_shape: 模型输入shape
      - --data_dir: 校验数据
      - --data_types: 数据类型
  
   量化后的模型存放路径为 `models/decoder_quant_deploy_model.onnx`。

3. 执行ATC命令。

  a、encoder模型转om
     ```
     atc --framework=5 \
        --model=segment-anything/models/encoder_quant_deploy_model.onnx \
        --output=segment-anything/models/encoder_quant \
        --input_format=NCHW \
        --input_shape="x:1,3,1024,1024" \
        --op_select_implmode=high_performance \
        --soc_version=Ascend${chip_name} \
        --log=error
     ```

     - 参数说明：
        -  --model：为ONNX模型文件。
        - --framework：5代表ONNX模型。
        - --output：输出的OM模型。
        - --input_format：输入数据的格式。
        - --input_shape：输入数据的shape。
        - --log：日志级别。
        - --soc_version：处理器型号。
        - --input_format：输入数据的格式。
        - --op_select_implmode:高性能模式。
     
     运行成功后在models目录下生成**encoder_quant.om**模型文件。

  b、decoder模型转om
     ```
     atc --framework=5 \
        --model=segment-anything/models/decoder_quant_deploy_model.onnx \
        --output=segment-anything/models/decoder_quant \
        --input_format=ND \
        --input_shape="image_embeddings:1,256,64,64;point_coords:1,-1,2;point_labels:1,-1;mask_input:1,1,256,256;has_mask_input:1" \
        --dynamic_dims="2,2;3,3;4,4;5,5;6,6;7,7;8,8;9,9" \
        --op_select_implmode=high_performance \
        --soc_version=Ascend${chip_name} \
        --log=error
     ```

     - 参数说明：
        -  --model：为ONNX模型文件。
        - --framework：5代表ONNX模型。
        - --output：输出的OM模型。
        - --input_format：输入数据的格式。
        - --input_shape：输入数据的shape。
        - --dynamic_dims:decoder输入分档参数设置
        - --log：日志级别。
        - --soc_version：处理器型号。
        - --input_format：输入数据的格式。
        - --op_select_implmode:高性能模式。
     
     运行成功后在models目录下生成**decoder_quant.om**模型文件。
        >**说明：**
     --dynamic_dims分档设置参数具体分为多少档根据实际情况可自行设置。

</details>

### 3.5 推理验证

1. 端到端推理。

   ```bash
   python3 sam_end2end_infer.py \
     --src-path data/demo.jpg \
     --save-path . \
     --encoder-model-path models/encoder_modify.om \
     --decoder-model-path models/decoder_modify.om \
     --input-point '[[500, 375], [1125, 625], [1520, 625]]' \
     --device-id 0
   ```

   参数说明：
  
   - src-path：图片数据路径。
   - save-path：SAM离线推理保存路径。
   - encoder-model-path：encoder模型路径。
   - decoder-model-path：decoder模型路径。
   - input-point：分割目标上的坐标（坐标数量根据带分割目标物体分割效果确定，不同的图片数据，不同的待分割目标坐标不一样）
   - device-id：NPU卡ID

   在线模型推理结果：

   ![](./assets/pth_truck_result.JPG)

   离线模型推理结果：

   ![](./assets/om_truck_result.JPG)

2. 性能验证。

   1. encoder 纯推理性能验证。

      ```bash
      python3 -m ais_bench --model models/encoder_modify.om --loop 100
      ```

      参数说明：
  
      - model: OM 模型。
      - loop: 循环次数。

   2. decoder 纯推理性能验证。

      这里以输入 3 个坐标为例。

      ```bash
      num_points=3
      python3 -m ais_bench \
        --model models/decoder_modify.om \
        --dymDims "image_embeddings:1,256,64,64;point_coords:1,${num_points},2;point_labels:1,${num_points};mask_input:1,1,256,256;has_mask_input:1" \
        --outputSize '1000,1000000' \
        --loop 100 
      ```

      参数说明：

      - model: om模型
      - outputSize:动态模型输出Size设置
      - auto_set_dymdims_mode：开启动态dims模式
      - loop: 循环次数
      - batchsize: 模型batch size

## 4. 模型推理性能 & 精度

| 芯片型号 | 模型 | Batch Size | 性能 |
| ---- | ---- | ---- | ---- |
| 310P3 | encoder | 1 | 4.15 fps |
| 310P3 | decoder | 1 | 464 fps |
