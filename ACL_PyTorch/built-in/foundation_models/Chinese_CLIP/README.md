# Chinese_CLIP模型推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Chinese_CLIP为CLIP模型的中文版本，使用大规模中文数据进行训练（~2亿图文对），本文档以**零样本图片分类任务**为例介绍离线推理步骤。


- 参考实现：

  ```
  url=https://github.com/OFA-Sys/Chinese-CLIP.git
  commit_id=2c38d03557e50eadc72972b272cebf840dbc34ea
  ```

- **参考配置**：

  ```
  clip_cn_vit-b-16
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 |   大小   |   数据类型      | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | image    | ${bs} x 3 x 224 x 224 | FLOAT32 | NCHW     |


- 输出数据

  | 输出数据 | 大小        | 数据类型 | 数据排布格式 |
  |-----------| -------- | -------- | ------------|
  | output  | ${bs} x 512 | FLOAT32  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 24.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN（+ MindIE）                                               | 8.0.RC1 | -                                                            |
| Python                                                       | 3.8.17   | -                                                            |
| PyTorch                                                      | 1.11.0 | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

说明：由于模型性能优化需要，需要安装与CANN包版本配套的MindIE。

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码。

   ```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-Pytorch/ACL_Pytorch/built-in/foundation/Chinese-CLIP
   ```

2. 获取第三方源码。

   ```shell
   # 请按顺序执行如下命令
   git clone https://github.com/OFA-Sys/Chinese-CLIP.git
   cd Chinese-CLIP
   git reset --hard 2c38d03557e50eadc72972b272cebf840dbc34ea
   
   pip3 install -r requirements.txt
   
   # 修改第三方源码推理适配部分
   patch -p0 < ../cn_clip.patch
   pip3 install -e .
   
   cd ..
   ```

3. 安装离线推理所需依赖。

   ```shell
   pip3 install -U pip && pip3 install -r requirements.txt
   ```
   
4. 安装改图工具 [AIT debug surgeon](https://gitee.com/ascend/ait/tree/master/ait/docs/debug/surgeon)，安装方式详见[一体化安装指导](https://gitee.com/ascend/ait/blob/master/ait/docs/install/README.md)。

   ```shell
   git clone https://gitee.com/ascend/ait.git
   # 1. git pull origin 更新最新代码 
   cd ait/ait
   # 2. 安装 ait 包
   pip install .
   # 3. 通过ait install 命令，安装surgeon组件
   ait install surgeon
   cd ../..
   ```

5. 设置环境变量。

   ```shell
   export PYTHONPATH=${PYTHONPATH}:$(pwd)/Chinese-CLIP/cn_clip
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      ```
      mkdir models
      ```

      下载[权重文件](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt) clip_cn_vit-b-16.pt 置于 models 目录下

   2. 导出onnx文件。

      1. 使用 Chinese-CLIP/cn_clip/deploy/pytorch_to_onnx.py 导出onnx文件。

         ```shell
         python3 Chinese-CLIP/cn_clip/deploy/pytorch_to_onnx.py \
         --model-arch ViT-B-16 \
         --pytorch-ckpt-path ./models/clip_cn_vit-b-16.pt \
         --save-onnx-path ./models/vit-b-16 \
         --context-length 512 \
         --convert-text \
         --convert-vision
         ```
         - 参数说明
              - --model-arch: 模型骨架
              - --pytorch-ckpt-path: Pytorch模型ckpt路径
              - --save-onnx-path: 输出ONNX格式模型的路径
              - --context-length：输入文本的padding length
              - --convert-vision: 指定是否转文本侧模型
              - --convert-vision: 指定是否转图像侧模型
         
         运行成功后，使用models目录下生成的 vit-b-16.txt.fp16.onnx 和 vit-b-16.img.fp16.onnx 文件进行后续操作。


   3. 使用 onnx-simplifier 简化 onnx 模型。


      1. 文本模型
    
            ```shell
            # export bs=20
            onnxsim models/vit-b-16.txt.fp16.onnx models/vit-b-16.txt.fp16.bs${bs}.sim.onnx --overwrite-input-shape "text:${bs},512"
            ```
    
      2. 图像模型
    
            ```shell
            # export bs=20
            onnxsim models/vit-b-16.img.fp16.onnx models/vit-b-16.img.fp16.bs${bs}.sim.onnx --overwrite-input-shape "image:${bs},3,224,224"
            ```

   4. 使用 opt_onnx.py 优化 onnx 模型。


      1. 文本模型
    
            ```shell
            # export bs=20
            python3 opt_txt_onnx.py models/vit-b-16.txt.fp16.bs${bs}.sim.onnx models/vit-b-16.txt.fp16.bs${bs}.opt.onnx
            ```
    
      2. 图像模型
    
            ```shell
            # export bs=20
            python3 opt_img_onnx.py \
            --input_file models/vit-b-16.img.fp16.bs${bs}.sim.onnx \
            --output_file models/vit-b-16.img.fp16.bs${bs}.opt.onnx \
            --model_config vit_base_patch16_224 \
            --use_flashattention
            ```
    
            - 参数说明：
              - --use_flashattention: 根据硬件形态选择是否添加此参数。

   5. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
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

      3. 执行ATC命令。
         ```shell
         # 例如 export bs=20 && export chip_name=310P3
         atc --framework=5 \
         --model=models/vit-b-16.txt.fp16.bs${bs}.opt.onnx \
         --output=models/vit-b-16.txt.fp16.bs${bs} \
         --input_format=ND \
         --input_shape="text:${bs},512" \
         --soc_version=Ascend${chip_name} \
         --log=error \
         --optypelist_for_implmode="Gelu" \
         --op_select_implmode=high_performance
         
         # 例如 export bs=20 && export chip_name=310P3
         atc --framework=5 \
         --model=models/vit-b-16.img.fp16.bs${bs}.opt.onnx \
         --output=models/vit-b-16.img.fp16.bs${bs} \
         --input_format=NCHW \
         --input_shape="image:${bs},3,224,224" \
         --soc_version=Ascend${chip_name} \
         --log=error \
         --optypelist_for_implmode="Sigmoid" \
         --op_select_implmode=high_performance \
         --insert_op_conf aipp.config \
         --enable_small_channel 1
         ```
         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input_format：输入数据的格式。
            - --input_shape：输入数据的shape。
            - --log：日志级别。
            - --soc_version：处理器型号。
            - --optypelist_for_implmode：指定算子。
            - --op_select_implmode：选择高性能/高精度模式，与 --optypelist_for_implmode 配合使用。
            - --insert_op_conf：AIPP配置文件。
            - --enable_small_channel：与 --insert_op_conf 配合使用。
         
         运行成功后，在 models 目录下生成 vit-b-16.img.fp16.${bs}.om 离线模型文件。

6. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 余弦相似度精度验证。

      ```
      python3 compare.py --img-model-path models/vit-b-16.img.fp16.bs${bs}.om --batch-size ${bs}
      ```
   
      - 参数说明：
   
        - --img-model-path：图像侧模型所在路径
        - --batch-size：批次大小
   
      运行成功后打印om推理结果与cpu推理结果的余弦相似度，确保精度正常。
   
   4. 性能验证。
   
      纯推理性能测试命令如下：
   
      ```shell
      # export bs=20
      python3 -m ais_bench --model models/vit-b-16.txt.fp16.bs${bs}.om --loop 50
      
      # export bs=20
      python3 -m ais_bench --model models/vit-b-16.img.fp16.bs${bs}.om --loop 50
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型纯推理性能数据参考如下数据。

- 文本侧模型：

| 芯片型号 | Input Shape | 单次推理耗时 |
| -------- | ----------- | ------------ |
| 310P3    | 1 x 512     | 5.96ms       |

- 图像侧模型：

| 芯片型号  | Batch Size | 单次推理耗时 |
|----------|------------|----------|
|  310P3  |       1       | 2.56ms       |
| 310P3 | 8 | 13.24ms |
| 310P3 | 16 | 24.35ms |
| 310P3 | 24 | 38.01ms |
| 310P3 | 32 | 52.02ms |
| 310P3 | 64 | 114.36ms |
