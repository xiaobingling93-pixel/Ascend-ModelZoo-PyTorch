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

BLIP 是一种多模态 Transformer 模型，提出了一种高效率利用噪声网络数据的方法，这种VLP框架可以灵活地在视觉理解任务上和生成任务上面迁移。本模型以下游任务Image Captioning为例进行推理。

- 参考实现：
  ```bash
    url=https://github.com/salesforce/BLIP.git
    commit_id=3a29b7410476bf5f2ba0955827390eb6ea1f4f9d
    model_name=model_base_caption_capfilt_large
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型             | 数据排布格式 |
  | -------- | -------- | -------------------- | ------------ |
  | image    |  bs x 3 x 384 x 384 |  FLOAT32   |  NCHW  |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | bs x seq_len | FLOAT32  | ND    |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.0  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 7.0.0 | -                                                            |
  | Python                                                       | 3.9   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


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
   获取transformers的文件路径
   ```bash
   python3 -c "import os;import transformers;print(os.path.join(os.path.dirname(transformers.__file__), 'generation/utils.py'))"
   ```
   对得到的`${path-of-utils.py}`文件进行适配修改
   ```bash
   patch ${path-of-utils.py} < utils.patch
   ```
   对开源仓进行适配修改，并将脚本放在BLIP开源仓路径下
   ```bash
   cd BLIP
   git apply ../blip.patch
   cp ../*.py ./
   ```

4. 安装昇腾统一推理工具（AIT）

   请访问[AIT代码仓](https://gitee.com/ascend/ait/tree/master/ait#ait)，根据readme文档进行工具安装。

   安装AIT时，可只安装需要的组件：benchmark和debug，其他组件为可选安装。
   ```bash
   git clone https://gitee.com/ascend/ait.git
   cd ait/ait
   ./install.sh --benchmark
   ./install.sh --surgeon
   cd ../../
   ```


## 准备数据集<a name="section183221994411"></a>
本模型采用COCO2014数据集的验证集（val）进行精度评估。请获取[coco官方数据集](http://images.cocodataset.org/zips/val2014.zip)并解压后将val2014文件夹放在当前文件夹下。目录结构如下：

```
BLIP
├── val2014
      ├── COCO_val2014_000000184613.jpg
      ├── COCO_val2014_000000562150.jpg
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

       请提前下载模型权重，以避免执行后面步骤时可能会出现下载失败。模型权重文件：[model_base_caption_capfilt_large.pth](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth)。


   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         python3 pth2onnx.py \
               --config ./configs/caption_coco.yaml \
               --pth_path ./model_base_caption_capfilt_large.pth \
               --output_dir blip_models
         ```
         - 参数说明：
            - --config：开源仓配置文件路径。
            - --pth_path：权重路径。如果权重路径不存在或未配置，则通过配置内的链接地址重新下载权重。
            - --output_dir：保存onnx模型的文件夹路径。

         在blip_models文件夹下获得visual_encoder.onnx和text_decoder.onnx文件。

      2. 优化ONNX文件。

         运行modify_onnx.py脚本。
         ```
         python3 modify_onnx.py --model blip_models/text_decoder.onnx --new_model blip_models/text_decoder_md.onnx
         ```
         - 参数说明：
            - --model：要修改的模型路径。
            - --new_model：修改后onnx模型的保存路径。

         获得blip_models/text_decoder_md.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
         # bs可使用参数配置，beam_num由config给定，默认为3。

         atc --framework=5 \
             --model=blip_models/visual_encoder.onnx \
             --output=blip_models/visual_encoder \
             --input_shape="image:${bs},3,384,384" \
             --soc_version=Ascend${chip_name}

         # dymdims 模式
         atc --framework=5 \
             --model blip_models/text_decoder_md.onnx \
             --output=blip_models/text_decoder \
             --input_shape="input_ids:${bs*beam_num},-1;attention_mask:${bs*beam_num},-1;encoder_hidden_states:${bs*beam_num},577,768" \
             --dynamic_dims="4,4;5,5;6,6;7,7;8,8;9,9;10,10;11,11;12,12;13,13;14,14;15,15;16,16;17,17;18,18;19,19" \
             --input_format=ND \
             --soc_version=Ascend${chip_name}
         # dymshape 模式
         atc --framework=5 \
             --model blip_models/text_decoder_md.onnx \
             --output=blip_models/text_decoder \
             --input_shape="input_ids:${bs*beam_num},-1;attention_mask:${bs*beam_num},-1;encoder_hidden_states:${bs*beam_num},577,768" \
             --input_format=ND \
             --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input\_format：输入数据的格式。
           - --input\_shape：输入数据的shape。
           - --dynamic_dims：输入数据的动态case。text_decoder的case的最小值为config中prompt的长度+1，最大值为config中max_length-1。
           - --soc\_version：处理器型号。

           运行成功后生成blip_models/visual_encoder.om和blip_models/text_decoder_{os}_{arch}.om模型文件。

2. 开始推理验证。


   1. 执行推理。

      ```bash
      python3 ascend_infer.py \
            --config ./configs/caption_coco.yaml \
            --caption_file blip_models/captions.json \
            --dataset_split val \
            --model_dir blip_models \
            --device 0 \
            --batch_size ${bs}
      ```

      - 参数说明：

         - --config：开源仓配置文件路径。
         - --caption_file：生成文本的保存路径。
         - --dataset_split：开源仓划分的数据集，val为验证集，test为测试集。
         - --model_dir：om模型的所在的文件夹路径。
         - --device：推理使用的npu芯片id。
         - --mode：decoder模型类型。默认为静态分档。
         - --batch_size：推理时数据的batch size。
         - --num_workers：数据处理的子进程个数，默认值为4，与开源仓一致。

      推理后的输出保存在blip_models/captions.json内。


   2. 精度验证。

      ```bash
      python3 ascend_infer.py \
            --config ./configs/caption_coco.yaml \
            --caption_file blip_models/captions.json \
            --dataset_split val \
            --evaluate \
            --evaluate_file blip_models/evaluate.json
      ```

      - 参数说明：

         - --config：开源仓配置文件路径。
         - --caption_file：生成文本的保存路径。
         - --dataset_split：开源仓划分的数据集，val为验证集，test为测试集。
         - --evaluate：开启精度验证功能。如果caption_file路径不存在，会自动进行推理并保存至caption_file。推理时参数设置同上。
         - --evaluate_file：精度结果保存路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 硬件型号 | Batch Size   | 数据集 | 性能 |
| --------- | ---------------- | ---------- | --------------- |
|  Atlas 300I Pro   |       32        |    coco-val2014   |     3.4 image/s    |

精度验证结果
```bash
Bleu_1: 0.788
Bleu_2: 0.639
Bleu_3: 0.507
Bleu_4: 0.400
METEOR: 0.308
ROUGE_L: 0.599
CIDEr: 1.325
SPICE: 0.238
```
