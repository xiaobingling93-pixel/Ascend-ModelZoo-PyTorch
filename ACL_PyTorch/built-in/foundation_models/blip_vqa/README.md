# blip模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)
    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

BLIP 是一种多模态 Transformer 模型，提出了一种高效率利用噪声网络数据的方法，这种VLP框架可以灵活地在视觉理解任务上和生成任务上面迁移。本模型以下游任务VQA为例进行推理。

- 参考实现：
  ```bash
    url=https://github.com/salesforce/BLIP.git
    commit_id=3a29b7410476bf5f2ba0955827390eb6ea1f4f9d
    model_name=model_base_vqa_capfilt_large
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型             | 数据排布格式 |
  | -------- | -------- | -------------------- | ------------ |
  | image    |  bs x 3 x 480 x 480 |  FLOAT32   |  NCHW  |
  | question |  bs x 35 |  int64   |  ND  |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | max_ids  | bs x 1 | FLOAT32  | ND    |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 24.1.RC2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN（+MindIE-RT）                                           | 8.0.RC2(1.0.RC2) | -                                                  |
  | Python                                                       | 3.10   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |
由于模型性能优化需要，需要安装与CANN包版本配套的MindIE。

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```bash
   git clone https://github.com/salesforce/BLIP.git
   ```

2. 安装依赖。
   ```bash
   pip3 install -r requirements.txt
   ```

3. 代码修改

   对开源仓进行适配修改，并将脚本放在BLIP开源仓路径下
   ```shell
   cd BLIP
   git apply ../blip.patch
   cp ../*.py ./
   ```

3. 安装昇腾统一推理工具（AIT）

   请访问[AIT代码仓](https://gitee.com/ascend/ait/tree/master/ait#ait)，根据readme文档进行工具安装。

   安装AIT时，可只安装需要的组件：benchmark和debug，其他组件为可选安装。
   ```bash
   git clone https://gitee.com/ascend/ait.git
   cd ait/ait
   chmod +x install.sh
   ./install.sh --benchmark
   ./install.sh --surgeon
   cd ../../
   ```


## 准备数据集<a name="section183221994411"></a>
本模型采用COCO2105数据集的测试集（test）进行精度评估。请获取[coco官方数据集](https://cocodataset.org/zips/test2015.zip)并解压后将test2015文件夹放在当前文件夹下。目录结构如下：

```
BLIP
├── test2015
   ├── COCO_val2015_000000262144.jpg
   ├── COCO_val2015_000000130190.jpg
   └── ...
```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取vocab.txt文件：请在[bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main)仓内下载vocab.txt文件，并放在bert-base-uncased目录下。

       ```bash
       mkdir bert-base-uncased
       mv vocab.txt bert-base-uncased/
       ```

       请提前下载模型权重，以避免执行后面步骤时可能会出现下载失败。模型权重文件：[model_base_vqa_capfilt_large.pth](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth)。


   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         python3 pth2onnx.py \
               --config ./configs/vqa.yaml \
               --pth_path ./model_base_vqa_capfilt_large.pth \
               --output_dir blip_models
         ```
         - 参数说明：
            - --config：开源仓配置文件路径。
            - --pth_path：权重路径。如果权重路径不存在或未配置，则通过配置内的链接地址重新下载权重。
            - --output_dir：保存onnx模型的文件夹路径。

         运行后生成文件
         ```
         ├── blip_models
            ├── visual_encoder.onnx
            ├── text_encoder.onnx
            ├── text_decoder_1.onnx
            └── text_decoder_2.onnx
         ```

      2. 优化ONNX文件。

         运行modify_onnx.py脚本。
         ```shell
         bs=32
         python3 modify_onnx.py --model_dir blip_models/ --batch_size ${bs}
         ```
         - 参数说明：
            - --model：要修改的模型路径。
            - --batch_size：推理时数据的batch size。

         运行后生成文件
         ```
         ├── blip_models
            ├── visual_encoder_md.onnx
            ├── text_encoder_md.onnx
            └── text_decoder_2_md.onnx
         ```

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```shell
         source /usr/local/Ascend/ascend-toolkit/set_env.sh

         # 如果安装了推理引擎算子包，需配置推理引擎路径
         source /usr/local/Ascend/mindie-rt/set_env.sh
         ```
         **说明：**
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
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

      3. 执行ATC命令。

         ```bash
         # k_test与配置文件一致，默认为128，k_testxbs=k_test*bs
         k_testxbs=4096

         atc --framework=5 \
             --model=blip_models/visual_encoder_md.onnx \
             --output=blip_models/visual_encoder \
             --input_shape="image:${bs},3,480,480" \
             --soc_version=Ascend${chip_name}

         atc --framework=5 \
             --model blip_models/text_encoder_md.onnx \
             --output=blip_models/text_encoder \
             --input_shape="input_ids:${bs},35;attention_mask:${bs},35;image_embeds:${bs},901,768" \
             --input_format=ND \
             --soc_version=Ascend${chip_name}

         atc --framework=5 \
             --model blip_models/text_decoder_1.onnx \
             --output=blip_models/text_decoder_1 \
             --input_shape="start_ids:${bs},1;question_states:${bs},35,768;question_atts:${bs},35" \
             --input_format=ND \
             --soc_version=Ascend${chip_name}
             
         atc --framework=5 \
             --model blip_models/text_decoder_2_md.onnx \
             --output=blip_models/text_decoder_2 \
             --input_shape="input_ids:${k_testxbs},8;input_atts:${k_testxbs},8;question_states:${k_testxbs},35,768;question_atts:${k_testxbs},35;target_ids:${k_testxbs},8" \
             --input_format=ND \
             --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input\_format：输入数据的格式。
           - --input\_shape：输入数据的shape。
           - --soc\_version：处理器型号。

         运行后生成文件
         ```
         ├── blip_models
            ├── visual_encoder.om
            ├── text_encoder.om
            ├── text_decoder_1.om
            └── text_decoder_2.om
         ```

2. 开始推理验证。


   1. 执行推理。

      ```bash
      python3 ascend_infer.py \
            --config ./configs/vqa.yaml \
            --result_file blip_models/results.json \
            --image_dir ./ \
            --model_dir blip_models \
            --device 0 \
            --batch_size ${bs}
      ```

      - 参数说明：

         - --config：开源仓配置文件路径。
         - --result_file：推理结果的保存路径。
         - --image_dir：数据图片路径，test2015文件夹所在路径。
         - --model_dir：om模型的所在的文件夹路径。
         - --device：推理使用的npu芯片id。
         - --batch_size：推理时数据的batch size。
         - --num_workers：数据处理的子进程个数，默认值为4，与开源仓一致。

      推理后的输出保存在blip_models/results.json内。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 硬件型号 | Batch Size   | 性能 |
| --------- | ---------------- | --------------- |
|  Atlas 300I Pro   |       32        |      7.9 data/s    |

