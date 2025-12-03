# SenseVoice + Vad 推理指导

- [SenseVoice + Vad 推理指导](#sensevoice--vad-推理指导)
  - [概述](#概述)
  - [推理环境准备](#推理环境准备)
  - [获取源码](#获取源码)
  - [数据集准备](#数据集准备)
  - [模型推理](#模型推理)
    - [模型转换](#模型转换)
    - [目录结构](#目录结构)
    - [执行推理命令](#执行推理命令)
  - [性能数据](#性能数据)
  - [精度数据](#精度数据)

## 概述

本文档参考[SenseVoice(ONNX)-推理指导](../../../built-in/audio/SenseVoice/README_onnx.md)，新增vad语音端点检测模型，用于检测音频中有效的语音片段，并支持输出timestamp（每个识别词对应音频中的时间）

- 版本说明：
  
  ```
  url=https://github.com/modelscope/FunASR
  commit_id=c4ac64fd5d24bb3fc8ccc441d36a07c83c8b9015
  ```

## 推理环境准备

**表 1**  版本配套表

| 配套                                                                          | 版本     | 环境准备指导                                                                                       |
| ----------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------- |
| 固件与驱动                                                                    | 25.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                                          | 8.2.RC1  | -                                                                                                  |
| Python                                                                        | 3.11.10  | -                                                                                                  |
| PyTorch                                                                       | 2.1.0    | -                                                                                                  |
| 说明：支持Atlas 300I Duo | \        | \                                                                                                  |

## 获取源码

1. 获取本仓源码

```shell
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/contrib/audio/SenseVoice
```

2. 安装依赖

```shell
pip3 install -r requirements.txt
```

3. 获取 `Pytorch`源码

```shell
git clone https://github.com/modelscope/FunASR.git
cd FunASR
git reset c4ac64fd5d24bb3fc8ccc441d36a07c83c8b9015 --hard
git apply ../diff.patch
cp ../export_onnx.py ./
cp ../om_infer.py ./
```

4. 安装aisbench工具
   参考[aisbench](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)安装aisbench工具
5. 安装msit工具
   参考[msit](https://gitcode.com/ascend/msit)安装工具中surgeon组件。
6. 获取权重

+ [SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall/files)
+ [speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)

## 数据集准备

* librispeech_asr_dummy数据集[下载地址](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/main)

## 模型推理

### 模型转换

1. 导出onnx模型

```shell
python3 export_onnx.py --model /path/to/SenseVoiceSmall
```

+ 参数说明
+ --model SenseVoiceSmall模型路径

脚本运行后会在权重目录下生成model.onnx文件

1. 修改onnx模型

```
# ${ModelZoo-PyTorch} 为modelzoo代码所在路径
cp ${ModelZoo-PyTorch}/ACL_PyTorch/built-in/audio/SenseVoice/modify_onnx.py ./
python3 modify_onnx.py \
--input_path=/path/to/SenseVoiceSmall/model.onnx \
--save_path=/path/to/SenseVoiceSmall/model_md.onnx
```

修改原始onnx模型。删除多余的domian，生成新的model_md.onnx模型

1. 使用 `ATC`工具将 `ONNX`模型转为 `OM`模型

配置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

执行ATC命令，利用npu-smi info命令获取芯片型号,填入soc_version参数中

```
atc --framework=5 --soc_version=Ascend${soc_version} --model /path/to/SenseVoiceSmall/model_md.onnx --output SenseVoice --input_shape="speech:1,-1,560;speech_lengths:1;language:1;textnorm:1"
```

在当前目录下生成动态模型SenseVoice_{arch}.om

### 目录结构

```text
📁 SenseVoice/
├──📄 diff.patch
├──📄 export_onnx.py
├──📄 modify_onnx.py
├──📄 requirements.txt
├── 📁Funasr/
|   |── 📄 ...
|   |── 📁 funasr/
|   |── 📁 speech_fsmn_vad_zh-cn-16k-common-pytorch/
|   |── 📁 SenseVoiceSmall /
|   |── 📁 librispeech_asr_dummy
|   |── 📄 senseVoice_{arch}.om
|   |── 📄 om_infer.py
```

### 执行推理命令

```
python3 om_infer.py \
--vad_path speech_fsmn_vad_zh-cn-16k-common-pytorch \
--model_path SenseVoiceSmall \
--om_path SenseVoice_{arch}.om \
--device 0 \
--input ./librispeech_asr_dummy/validation-00000-of-00001.parquet \
--output_timestamp
```

- 参数说明
- vad_path: vad模型权重路径
- model_path: SenseVoice模型权重路径
- om_model: om模型路径
- device: npu芯片id，默认使用0卡
- input: librispeech_asr_dummy数据集文件的路径
- output_timestamp 输出时间戳

执行后会打印在该数据集下的转录比和WER率

## 性能数据

| 模型             | 数据集                | 芯片         | 转录比 | T4转录比 |
| ---------------- | --------------------- | ------------ | ------ | ---------- |
| SenseVoice + Vad | librispeech_asr_dummy | 300I Duo单芯 | 23     | 35         |

## 精度数据

| 模型             | 数据集                | 芯片         | WER   | T4 WER |
| ---------------- | --------------------- | ------------ | ----- | ------- |
| SenseVoice + Vad | librispeech_asr_dummy | 300I Duo单芯 | 0.083 | 0.083   |

