# Paraformer(TorchAir)-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******

# 概述
Paraformer是阿里达摩院语音团队提出的一种高效的非自回归端到端语音识别框架。本项目为Paraformer中文通用语音识别模型，采用工业级数万小时的标注音频进行模型训练，保证了模型的通用识别效果。模型可以被应用于语音输入法、语音导航、智能会议纪要等场景。

- 版本说明：
  ```
  url=https://github.com/modelscope/FunASR
  commit_id=c4ac64fd5d24bb3fc8ccc441d36a07c83c8b9015
  model_name=paraformer
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            | 版本           | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    |--------------| ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 25.3.rc1       | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 8.3.RC1        | -                                                                                                   |
  | Python                                                          | 3.11         | -                                                                                                     |
  | PyTorch                                                         | 2.1.0        | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.1.0.post13 | -                                                                                                     |
  | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \            | \                                                                                                     |


# 快速上手

## 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/Paraformer
   ```
   
1. 安装依赖  
   ```bash
   pip3 install -r requirements.txt
   ```

2. 获取模型仓源码
   ```bash
   git clone https://github.com/modelscope/FunASR.git
   cd FunASR
   git reset --hard c4ac64fd5d24bb3fc8ccc441d36a07c83c8b9015
   git apply ../diff_Funasr.patch
   pip3 install -e ./
   cd ..
   ```

3. 下载模型权重

    - [Paraformer-large-长音频版](https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)

    - [Paraformer-large-热词版模型](https://www.modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/files)

    - [VAD模型](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/files) 作为前处理模块，切分长语音

    - [punc模型](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/files) 作为后处理模块，为paraformer转录结果添加标点符号

4. 下载数据集
    - [AISHELL-1](https://www.openslr.org/33/)


5. 完整下载后的文件目录树如下

    ```shell
    Paraformer
    ├── FunASR      // 从开源代码仓下载的文件夹
    ├── data_aishell   // AISHELL-1数据集
    ├── speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404   // Paraformer热词版模型权重
    ├── speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch // Paraformer长音频版权重
    ├── speech_fsmn_vad_zh-cn-16k-common-pytorch // VAD模型权重
    ├── punc_ct-transformer_cn-en-common-vocab471067-large  // punc模型权重
    ├── diff_Funasr.patch
    ├── infer.py           // 本仓库提供的自定义推理脚本
    ├── test_performance.py  // 本仓库提供的性能测试脚本
    ├── test_accuracy.py     // 本仓提供的精度测试脚本
    ├── torchair_auto_model.py
    ├── README.md
    └── requirements.txt
    ```

## 模型推理

1. 样本测试

    ```bash
    # 热词版
    python3 infer.py \
      --model_path=./speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404 \
      --data=./speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/example/asr_example.wav \
      --hotwords="魔搭"
    # 长音频版
    python3 infer.py \
      --model_path=./speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
      --model_vad=./speech_fsmn_vad_zh-cn-16k-common-pytorch \
      --model_punc=./punc_ct-transformer_cn-en-common-vocab471067-large \
      --data=./speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav
    ```
    - 参数说明：
      - model_path: paraformer模型权重路径
      - model_vad: vad模型权重路径，默认为None，表示不使用VAD模型
      - model_punc: punc模型权重路径，默认为None，表示不使用punc模型
      - data: 模型输入文件，默认为长序列paraformer文件中的asr_example.wav
      - hotwords: 语音热词，只有在使用热词版paraformer时该参数才有效，默认为None
      - device: npu芯片id，默认为0
      - batch_size: 模型输入batch size，如果输入数量不是batch_size的整倍数，会在前处理时pad到batch size的整倍数，默认为1
      - warmup: warm up次数
  
    推理脚本以计算单用例音频输出结果为例，推理后将打屏推理结果

2. 性能测试 执行以下命令对**Paraformer长序列版**进行在全量AISHELL-1测试集上的性能测试，该脚本仅针对Paraformer模型进行测试，不包含vad和punc模块，batch_size参数用于控制同时处理的最大音频数量（例如设置为64，则会在sample_path下同时读取64个音频，并组合成一个输入进行处理，若不足会补全）
    ```bash
    python3 test_performance.py \
    --model_path=./speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --batch_size=64 \
    --data_path=./path/to/AISHELL-1/wav/test \
    --result_path=./aishell_test_result.txt \
    --warm_up=3
    ```
    - 参数说明：
      - --model_path: paraformer模型权重路径
      - --batch_size: 模型输入batch size，如果输入数量不是batch_size的整倍数，会在前处理时pad到batch size的整倍数，默认为64
      - data_path: AISHELL-1测试集音频所在路径，测试脚本会递归查找该路径下的所有音频文件
      - --result_path：测试音频的推理结果的保存路径

3. 精度测试 该精度测试只针对**Paraformer长序列版**需要首先执行完性能测试，而后利用性能测试保存到result_path的结果进行精度验证，执行如下命令
    ```bash
    python3 test_accuracy.py \
    --result_path=./aishell_test_result.txt \
    --ref_path=/path/to/AISHELL-1/transcript/aishell_transcript_v0.8.txt
    ```

# 模型推理性能&精度

| 模型     | 硬件 | 数据集 | batch size | paraformer推理性能(转录比)  | 竞品(A10)性能(转录比) | 精度CER | 竞品(A10)CER|
|----------|------|-------------------|----------------|------------|------------|-----|-----|
| Paraformer长序列版 |Atlas 800I A2| AISHELL-1 | 64| 840 | 513 | 0.0198 | 0.019873|
|Paraformer长序列版 | Atlas 300I DUO| AISHELL-1 | 16| 180 |513 | 0.0204 |0.019873|

