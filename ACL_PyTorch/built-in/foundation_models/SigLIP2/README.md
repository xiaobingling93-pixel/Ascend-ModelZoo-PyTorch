
# SigLIP2模型离线推理指导


- [概述](#概述)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码](#获取源码)
  - [模型推理](#模型推理)

- [模型推理性能&精度](#模型推理性能&精度)

  ******


# 概述

SigLIP 是改进损失函数后的 CLIP 多模态模型，可用于零样本图像分类、图文检索等任务。
SigLIP 2模型在SigLIP的原始训练目标基础上，增加了额外的目标，以提升语义理解、定位和密集特征提取能力。SigLIP 2兼容SigLIP，用户只需要更换模型权重和tokenizer，可以按照相同的步骤进行模型推理。
本文档以 siglip2-so400m-patch14-384 为例介绍离线推理步骤。
在线推理可以参考TorchAir目录下的指导。


## 输入输出数据

- 输入数据

  | 输入数据 |   大小   |   数据类型      | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
   | text   | labelnums x 64 | INT32 | ND     |
  | image    | bs x 3 x 384 x 384 | FLOAT32 | NCHW     |

- 输出数据

  | 输出数据 | 大小        | 数据类型 | 数据排布格式 |
  |-----------| -------- | -------- | ------------|
  | text_feature  | labelnums x 1152 | FLOAT32  | ND           |
  | image_feature  | bs x 1152 | FLOAT32  | ND           |
# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 25.2.0 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                               | 8.2.RC1 | -                                                            |
| Python                                                       | 	3.11  | -                                                            |
| PyTorch                                                      | 2.4.0 | -                                                            |
| Ascend Extension PyTorch                                                     | 2.4.0.post2 | -                                                            |
| 说明：Atlas 800T A2 推理卡请以CANN版本选择实际固件与驱动版本 | \       | \                                                            |


# 快速上手

## 获取源码

1. 获取本仓源码。

   ```shell
   git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/foundation_models/SigLIP2
   ```


2. 安装离线推理所需依赖。

   ```shell
   pip3 install -U pip && pip3 install -r requirements.txt
   ```

## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      ```
      mkdir models
      ```

      下载[权重文件](https://huggingface.co/google/siglip2-so400m-patch14-384)siglip2-so400m-patch14-384 置于 models 目录下。

   2. 导出onnx文件。

         ```shell
         python3 pth2onnx.py \
         --pytorch_ckpt_path models/siglip2-so400m-patch14-384 \
         --save_onnx_path models/ \
         --convert_text \
         --convert_vision
         ```
         - 参数说明
              - --pytorch_ckpt_path: Pytorch模型权重路径
              - --save_onnx_path: 输出ONNX格式模型的路径
              - --convert_text: 指定是否转文本侧模型
              - --convert_vision: 指定是否转图像侧模型
         
         运行成功后，使用models目录下生成的 siglip2_text_encoder.onnx 和 siglip2_vision_encoder.onnx 文件进行后续操作。

   3. 使用ATC工具将ONNX模型转OM模型。

         1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

         2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend${chip_name} （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip              | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0      ${chip_name}       | OK              | 102.0        49                0    / 0              |
         | 0                 | 0000:82:00.0    | 0            0    / 0          4130 / 65536
         |
         +===================+=================+======================================================+
         | 1       ${chip_name}      | OK              | 97.4         49                0    / 0              |
         | 0                 | 0000:89:00.0    | 0            0    / 0          3622 / 65536                            |
         +===================+=================+======================================================+
         ```

         3. 执行ATC命令。
         ```shell
         # 例如 export labelnums=1000 && export chip_name=XXX
         atc --model=models/siglip2_text_encoder.onnx \
         --framework=5 \
         --output=models/siglip2_text_encoder_labelnums${labelnums} \
         --input_format=ND \
         --input_shape="input_ids:${labelnums},64" \
         --log=error \
         --soc_version=Ascend${chip_name} 

         # 例如 export bs=24 && export chip_name=XXX 
         atc --model=models/siglip2_vision_encoder.onnx \
         --framework=5 \
         --output=models/siglip2_vision_encoder_bs${bs} \
         --input_format=NCHW \
         --input_shape="image:${bs},3,384,384" \
         --log=error \
         --soc_version=Ascend${chip_name}
         ```
         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input_format：输入数据的格式。
            - --input_shape：输入数据的shape。
            - --log：日志级别。
            - --soc_version：处理器型号。
         
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 性能验证。
   
      纯推理性能测试命令如下：
   
      ```shell
      # export labelnums=1000
      python3 -m ais_bench --model models/siglip2_text_encoder_labelnums${labelnums}.om --loop 50
      
      # export bs=24
      python3 -m ais_bench --model models/siglip2_vision_encoder_bs${bs}_linux_aarch64.om --loop 50
      ```
    
    3. 精度验证。
       本模型使用[ImageNet ILSVRC 2012](https://www.image-net.org/challenges/LSVRC/index.php)验证集进行推理测试。 
       1. 获取原始数据集。（解压命令参考tar -xvf  \*.tar与 unzip \*.zip）
         用户自行获取数据集后，上传数据集并解压。数据集目录结构如下所示：

       ```
       ImageNet/
       |-- val
       |   |-- ILSVRC2012_val_00000001.jpeg
       |   |-- ILSVRC2012_val_00000002.jpeg
       |   |-- ILSVRC2012_val_00000003.jpeg
       |   ...
       |-- val_label.txt
       ...
       ```
       2. 下载[标签集合文件](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)命名为imagenet1000_clsidx_to_labels.txt置于SigLIP2目录下。
       3. 数据预处理，将原始数据集转换为模型的输入数据。
        ```shell
        python3 preprocess.py --data_dir ImageNet/val \
        --image_save_dir image_save_dir \
        --text_save_dir text_save_dir \
        --pytorch_ckpt_path models/siglip2-so400m-patch14-384 \
        --classnames_file imagenet1000_clsidx_to_labels.txt
        ```
        - 参数说明：
		    -  --data_dir：数据集所在路径。
		    -  --image_save_dir：生成的image数据集二进制文件的路径。
		    -  --text_save_dir：生成的标签二进制文件的路径。
		    -  --pytorch_ckpt_path: Pytorch模型ckpt路径。
          -  --classnames_file：标签集合文件。
       4. 执行推理。
        ```shell
        # 推理得到图像特征
       python3 -m ais_bench \
        --model models/siglip2_vision_encoder_bs${bs}_linux_aarch64.om \
        --batchsize ${bs} \
        --input image_save_dir \
        --output image_feature_result \
        --output_dirname result_bs${bs} \
        --outfmt TXT

        # 推理得到文本特征
        python3 -m ais_bench \
        --model models/siglip2_text_encoder_labelnums${labelnums}.om \
        --batchsize ${labelnums} \
        --input text_save_dir \
        --output text_feature_result \
        --output_dirname result_labelnums${labelnums} \
        --outfmt TXT
        ```
        - 参数说明：
	       -  --model：om模型路径。
		    -  --batchsize：批次大小。
	       -  --input：输入数据所在路径。
	       -  --output：推理结果输出路径。
	       -  --output_dirname：推理结果输出子文件夹。
	       -  --outfmt：推理结果输出格式。
       5. 后处理。
       用脚本与数据集真值标签val_label.txt比对，打屏Accuracy数据。

        ```shell
        python3 postprocess.py --text_feature_result text_feature_result/result_labelnums${labelnums} \
        --image_feature_result image_feature_result/result_bs${bs} \
        --gt_file ImageNet/val_label.txt \
        --label_nums ${labelnums} \
        --pytorch_ckpt_path models/siglip2-so400m-patch14-384
        ```
        - 参数说明：

	        -  --text_feature_result：文本特征所在路径。
	        -  --image_feature_result：图像特征所在路径。
	        -  --gt_file：真值标签文件所在路径。
           -  --label_nums：使用的数据集的标签类别总数。
	        -  --pytorch_ckpt_path：Pytorch模型ckpt路径。

# 模型推理性能&精度

调用ACL接口推理计算，模型纯推理性能数据参考如下数据。

- 文本侧模型：

| 硬件形态     | Input Shape | tps |
|:-----------|:------------|:------------:|
| Atlas 800T A2 (8*64G) 单卡    | 1000 x 64  |  1262.35    |

- 图像侧模型：

| 硬件形态    | Input Shape | fps |
|:-----------|:------------|:------------:|
| Atlas 800T A2 (8*64G) 单卡  | 24 x 3 x 384 x 384        |  81.48    |

模型OM部署的推理精度数据参考如下数据。

| 硬件形态 |  数据集|   精度 |  参考精度(GPU A10)|
|:------:|:------:|:----------:|:---------:|
|   Atlas 800T A2 (8*64G) 单卡 | ImageNet| top1: 74.07|top1: 74.10|