# BLIP 模型 VQA 任务推理指导

BLIP 是一种多模态 Transformer 模型，提出了一种高效率利用噪声网络数据的方法，这种 VLP 框架可以灵活地在视觉理解任务上和生成任务上面迁移。本模型以下游任务 VQA 为例进行推理。

```
url=https://github.com/salesforce/BLIP.git
commit_id=3a29b7410476bf5f2ba0955827390eb6ea1f4f9d
model_name=model_base_vqa_capfilt_large
```

## 1. 模型输入输出数据规格

1. visual_encoder

   输入数据：

   | 输入数据 | 大小 | 数据类型 | 数据排布格式 |
   | ---- | ---- | ---- | ---- |
   | image | bs x image_size x image_size x 3 | uint8 | NHWC |

   输出数据：

   | 输出数据 | 大小 | 数据类型 | 数据排布格式 |
   | ---- | ---- | ---- | ---- |
   | image_embeds | bs x 901 x 768 | float32 | ND |

2. text_encoder

   输入数据：

   | 输入数据 | 大小 | 数据类型 | 数据排布格式 |
   | ---- | ---- | ---- | ---- |
   | input_ids | bs x question_seq_len | int64 | NCHW |
   | attention_mask | bs x question_seq_len | int64 | NCHW |
   | image_embeds | bs x 901 x 768 | float32 | NCHW |

   输出数据：

   | 输出数据 | 大小 | 数据类型 | 数据排布格式 |
   | ---- | ---- | ---- | ---- |
   | logits | bs x question_seq_len x 768 | float32 | ND |

3. text_decoder（rank 推理模式）

   - text_decoder_rank_1

     输入数据：

     | 输入数据 | 大小 | 数据类型 | 数据排布格式 |
     | ---- | ---- | ---- | ---- |
     | start_ids | bs x 1 | int64 | NCHW |
     | question_states | bs x question_seq_len x 768 | float32 | NCHW |
     | question_atts | bs x question_seq_len | int64 | NCHW |

     输出数据：

     | 输出数据 | 大小 | 数据类型 | 数据排布格式 |
     | ---- | ---- | ---- | ---- |
     | logits | bs x tokenizer_len | float32 | ND |

   - text_decoder_rank_2

     输入数据：

     | 输入数据 | 大小 | 数据类型 | 数据排布格式 |
     | ---- | ---- | ---- | ---- |
     | input_ids | (bs * k_test) x 8 | int64 | NCHW |
     | input_atts | (bs * k_test) x 8 | int64 | NCHW |
     | question_states | (bs * k_test) x question_seq_len x 768 | float32 | NCHW |
     | question_atts | (bs * k_test) x question_seq_len | float32 | NCHW |
     | target_ids | (bs * k_test) x 8 | int64 | NCHW |

     输出数据：

     | 输出数据 | 大小 | 数据类型 | 数据排布格式 |
     | ---- | ---- | ---- | ---- |
     | output | (bs * k_test) | float32 | ND |

4. text_decoder（generate 推理模式）

   输入数据：

   | 输入数据 | 大小 | 数据类型 | 数据排布格式 |
   | ---- | ---- | ---- | ---- |
   | input_ids | bs x answer_seq_len | int64 | NCHW |
   | attention_mask | bs x answer_seq_len | int64 | NCHW |
   | encoder_hidden_states | bs x question_seq_len x 768 | float32 | ND |
   | encoder_attention_mask | bs x question_seq_len | float32 | ND |

   输出数据：

   | 输出数据 | 大小 | 数据类型 | 数据排布格式 |
   | ---- | ---- | ---- | ---- |
   | logits | bs x answer_seq_len x tokenizer_len | float32 | ND |

## 2. 环境搭建

1. 安装固件与驱动、CANN、MindIE、Python

   **表 1**  版本配套表

   | 配套 | 版本 |
   | ---- | ---- |
   | 固件与驱动 | 24.1.RC3 |
   | CANN | 8.0.RC3 |
   | MindIE | 1.0.RC3 |
   | Python | 3.10.14 |

   请参考 [Pytorch 框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)。

2. 安装 msit 的 debug surgeon

   请参考 [msit 安装指导](https://gitee.com/ascend/msit/blob/master/msit/docs/install/README.md)，安装 debug surgeon 组件即可。

3. 安装 ais_bench

   请参考 [ais_bench 安装指导](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#%E5%B7%A5%E5%85%B7%E5%AE%89%E8%A3%85)。

4. 获取 ModelZoo 源码

   ```bash
   https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/foundation_models/blip_vqa
   ```

5. 安装第三方包

   ```bash
   pip3 install -r requirements.txt
   ```

6. 获取 BLIP 源码

   ```bash
   git clone https://github.com/salesforce/BLIP.git
   cd BLIP
   git reset --hard 3a29b7410476bf5f2ba0955827390eb6ea1f4f9d
   ```

7. 应用 patch

   ```bash
   git apply ../blip.patch
   patch $(python3 -c "import os; import transformers; print(os.path.join(os.path.dirname(transformers.__file__), 'generation/utils.py'))") < ../utils.patch
   ```

## 3. 准备权重、词表、数据集

1. 准备权重

   ```bash
   wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth
   ```

2. 准备词表

   ```bash
   mkdir bert-base-uncased
   cd bert-base-uncased
   wget https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt?download=true
   cd ..
   ```

3. 准备数据集

   ```bash
   wget https://images.cocodataset.org/zips/test2015.zip
   unzip test2015.zip
   ```

   解压后目录结构如下：

   ```
   test2015
   ├── COCO_val2015_000000262144.jpg
   ├── COCO_val2015_000000130190.jpg
   └── ...
   ```

## 4. 设置推理模式

BLIP 模型 VQA 任务有 rank 和 generate 两种推理模式，两种推理模式的回答生成方式不同。

- 在 rank 模式下，模型从一个预定义的候选答案集合中选择最有可能的答案。具体来说，模型会对这些候选答案进行评分，然后根据评分的高低排序，最后选择得分最高的答案作为输出。
- 在 generate 模式下，模型直接生成一个答案，而不是从预定义的集合中选择。模型通过自回归的方式逐字（或逐词）生成答案，直到生成完整的回答。

在执行后续步骤前，请根据需要选择一个推理模式。

设置 rank 推理模式

```bash
infer_mode='rank'
```

或设置 generate 推理模式

```bash
infer_mode='generate'
```

## 5. 模型转换

### 5.1 导出 ONNX 模型

```bash
python3 ../pth2onnx.py \
  --config configs/vqa.yaml \
  --infer_mode "${infer_mode}" \
  --pth_path model_base_vqa_capfilt_large.pth \
  --output_dir ascend_models
```

rank 推理模式运行后生成文件：

```
ascend_models
├── visual_encoder.onnx
├── text_encoder.onnx
├── text_decoder_rank_1.onnx
└── text_decoder_rank_2.onnx
```

generate 推理模式运行后生成文件：

```
ascend_models
├── visual_encoder.onnx
├── text_encoder.onnx
└── text_decoder_generate.onnx
```

参数说明：

- --config：开源仓配置文件路径，默认为 configs/vqa.yaml。
- --infer_mode：推理模式，可选 rank 或 generate，默认为 rank。
- --pth_path：权重路径，默认为 model_base_vqa_capfilt_large.pth。如果权重路径不存在或未配置，则通过配置内的链接地址重新下载权重。
- --output_dir：保存 ONNX 模型的路径，默认为 ascend_models。

### 5.2 使用 onnxsim 固定 shape 和简化 ONNX 模型

```bash
bs=32 # 取自开源配置文件 https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/configs/vqa.yaml#L12
bash ../fix_shape_and_simplify.sh "${infer_mode}" "${bs}"
```

rank 推理模式运行后生成文件：

```
ascend_models
├── visual_encoder_sim.onnx
├── text_encoder_sim.onnx
├── text_decoder_rank_1_sim.onnx
└── text_decoder_rank_2_sim.onnx
```

generate 推理模式运行后生成文件：

```
ascend_models
├── visual_encoder_sim.onnx
├── text_encoder_sim.onnx
└── text_decoder_generate_sim.onnx
```

### 5.3 运行改图脚本，修改 ONNX 模型以适配昇腾芯片

```bash
python3 ../modify_onnx.py --infer_mode "${infer_mode}" --model_dir ascend_models
```

rank 推理模式运行后生成文件：

```
ascend_models
├── visual_encoder_md.onnx
├── text_encoder_md.onnx
├── text_decoder_rank_1_md.onnx
└── text_decoder_rank_2_md.onnx
```

generate 推理模式运行后生成文件：

```
ascend_models
├── visual_encoder_md.onnx
└── text_encoder_md.onnx
```

参数说明：

- --infer_mode：推理模式，可选 rank 或 generate，默认为 rank。
- --model_dir：存放模型的路径，默认为 ascend_models。

### 5.4 使用 ATC 工具将 ONNX 模型转为 OM 模型

1. 设置环境变量

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/mindie-rt/set_env.sh
   ```

   **说明：** 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

2. 执行命令查看芯片名称（${chip_name}）

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

3. 执行 atc 命令

   ```bash
   bash ../atc.sh "${infer_mode}" "${bs}" "${chip_name}"
   ```

   rank 推理模式运行后生成文件：

   ```
   ascend_models
   ├── visual_encoder_md.om
   ├── text_encoder_md.om
   ├── text_decoder_rank_1_md.om
   └── text_decoder_rank_2_md.om
   ```

   generate 推理模式运行后生成文件：

   ```
   ascend_models
   ├── visual_encoder_md.om
   ├── text_encoder_md.om
   └── text_decoder_generate.om
   ```

   参数说明：

   - --framework：原始框架类型。
   - --model：原始模型文件路径与文件名。
   - --output：存放转换后的离线模型的路径以及文件名。
   - --insert_op_conf：插入算子的配置文件路径与文件名，例如aipp预处理算子。
   - --enable_small_channel：是否使能small channel的优化，使能后在channel<=4的首层卷积会有性能收益。
   - --input_format：输入数据格式。
   - --input_shape：模型输入数据的shape。
   - --dynamic_dims：设置ND格式下动态维度的档位。适用于执行推理时，每次处理任意维度的场景。
   - --soc_version：模型转换时指定芯片版本。

   更多参数说明请参考 [ATC 参数概览](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha001/devaids/auxiliarydevtool/atlasatc_16_0039.html)（如果链接失效，请从 [CANN 社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition) 查找目录「开发工具 > ATC 工具 > 参数说明 > 参数概览」）。

## 6. 推理

### 6.1 ais_bench 纯推理

1. visual_encoder

   ```bash
   python3 -m ais_bench --model ascend_models/visual_encoder_md.om --loop 100
   ```

2. text_encoder

   ```bash
   python3 -m ais_bench --model ascend_models/text_encoder_md.om --loop 100
   ```

3. text_decoder（rank 推理模式）

   - text_decoder_rank_1

     ```bash
     python3 -m ais_bench --model ascend_models/text_decoder_rank_1_md.om --loop 100
     ```

   - text_decoder_rank_2

     ```bash
     python3 -m ais_bench --model ascend_models/text_decoder_rank_1_md.om --loop 100
     ```

4. text_decoder（generate 推理模式）

   ```bash
   for answer_seq_len in {1..10}; do
     python3 -m ais_bench \
       --model ascend_models/text_decoder_generate_sim.om \
       --dymDims "input_ids:${bs},${answer_seq_len};attention_mask:${bs},${answer_seq_len};encoder_hidden_states:${bs},${question_seq_len},768;encoder_attention_mask:${bs},${question_seq_len}" \
       --loop 100
   done
   ```

5. 参数说明

   - --model：需要进行推理的OM离线模型文件。
   - --dymDims：动态维度参数，指定模型输入的实际Shape。如ATC模型转换时，设置 --input_shape="data:1,-1;img_info:1,-1" --dynamic_dims="224,224;600,600"，dymDims参数可设置为：--dymDims "data:1,600;img_info:1,600"。
   - --loop：推理次数。默认值为1，取值范围为大于0的正整数。 profiler参数配置为true时，推荐配置为1。

   更多参数说明请参考 [ais_bench 参数说明](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)。

### 6.2 数据集端到端推理

```bash
python3 ../ascend_infer.py \
  --config configs/vqa.yaml \
  --batch_size "${bs}" \
  --infer_mode "${infer_mode}" \
  --model_dir ascend_models \
  --image_dir . \
  --result_file ascend_infer_results.json \
  --device 0
```

rank 推理模式运行后生成文件 ascend_infer_results.json，打屏端到端推理平均性能。

generate 推理模式运行后生成文件 ascend_infer_results.json。

> 说明：在 generate 推理模式下，不同的测试用例生成的 answer token 数量不一致，text_decoder_generate 执行次数不一致，需要的端到端耗时不一致，因此 generate 推理模式的端到端平均性能没有意义。

参数说明：

- --config：开源仓配置文件路径，默认为 configs/vqa.yaml。
- --batch_size：推理时数据的 batch size，要与模型的 batch size 保持一致。
- --infer_mode：推理模式，可选 rank 或 generate，默认为 rank。
- --model_dir：存放模型的路径，默认为 ascend_models。
- --image_dir：图片数据集 test2015 所在路径，默认为当前路径 `.`。
- --result_file：推理结果的保存路径，默认为 ascend_infer_results.json。
- --num_workers：数据处理的子进程个数，默认为 4，与开源仓一致。
- --device：推理使用的 NPU 芯片 ID，默认为 0。

## 7. 模型推理性能

| 模型 | 硬件型号 | Batch Size | Answer Sequence Length | 性能 |
| ---- | ---- | ---- | ---- | ---- |
| rank 推理模式端到端 | Atlas 300I Pro | 32 | 不涉及 | 9.87 data/s |
| visual_encoder | Atlas 300I Pro | 32 | 不涉及 | 87.93 fps |
| text_encoder | Atlas 300I Pro | 32 | 不涉及 | 729.18 fps |
| text_decoder_rank_1 | Atlas 300I Pro | 32 | 不涉及 | 4539.00 fps |
| text_decoder_rank_2 | Atlas 300I Pro | 32 | 不涉及 | 1708.29 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 1 | 4331.59 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 2 | 1237.96 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 3 | 1230.88 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 4 | 1232.55 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 5 | 1225.12 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 6 | 1219.90 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 7 | 1173.03 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 8 | 1102.35 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 9 | 1196.36 fps |
| text_decoder_generate | Atlas 300I Pro | 32 | 10 | 1190.50 fps |
