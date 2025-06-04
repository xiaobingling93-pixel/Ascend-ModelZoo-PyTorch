# SenseVoice(ONNX)-推理指导

- [SenseVoice(ONNX)-推理指导](#sensevoiceonnx-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
    - [1 模型转换](#1-模型转换)
    - [2 开始推理验证](#2-开始推理验证)
    - [3 性能](#3-性能)
- [公网地址说明](#公网地址说明)

******

# 概述
SenseVoice作为一款专注于高精度多语言语音识别的模型，其独特之处在于其广泛的语言覆盖、强大的情感辨识能力以及高效的推理性能。该模型基于超过40万小时的多样化语音数据训练而成，能够支持超过50种语言的识别，展现出卓越的跨语言识别能力。与市场上其他主流模型相比，SenseVoice在识别精度上实现了显著提升，特别是在复杂场景下的表现尤为出色。

- 版本说明：
  ```
  url=https://github.com/FunAudioLLM/SenseVoice
  commit_id=de00f2b
  model_name=SenseVoice
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 24.0.RC3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  8.0.RC3 | -                                                                                                   |
  | Python                                                          |  3.8.20 | -                                                                                                     |
  | PyTorch                                                         | 2.1.0 | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.1.0.post8 | -                                                                                                     |
  | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/SenseVoice
   ```

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/FunAudioLLM/SenseVoice
   cd SenseVoice
   git reset --hard de00f2b
   mv ../requirements.txt ./
   mv ../export_onnx.py ./
   mv ../modify_onnx.py ./
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```
   
3. 安装msit工具
   
   参考[msit](https://gitee.com/ascend/msit)安装工具中的benchmark和surgen组件。
   

4. 获取权重数据

   从 https://modelscope.cn/models/iic/SenseVoiceSmall 获取所有权重文件，放在新建目录SenseVoiceSmall内

## 模型推理

### 1 模型转换

1. 导出onnx模型
```
python3 export_onnx.py
```
脚本运行后会在权重目录下生成model.onnx文件

1. 修改onnx模型
```
python3 modify_onnx.py --input_path=./SenseVoiceSmall/model.onnx --save_path=./SenseVoiceSmall/model_md.onnx
```
修改原始onnx模型。删除多余的domian，生成新的model_md.onnx模型

   
1. 使用`ATC`工具将`ONNX`模型转为`OM`模型  

配置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

执行ATC命令，利用npu-smi info命令获取芯片型号,填入soc_version参数中


```
atc --framework=5 --soc_version=Ascend${soc_version} --model ./SenseVoiceSmall/model_md.onnx --output SenseVoice --input_shape="speech:1,-1,560;speech_lengths:1;language:1;textnorm:1"
```
在当前目录下生成动态模型SenseVoice_{arch}.om

### 2 开始推理验证

移动推理py文件到源码目录内
```
mv ../infer_onnx.py ./
```
执行推理命令

```
python3 infer_onnx.py --model_path=SenseVoiceSmall --om_path=SenseVoice_{arch}.om --device=0 --input="./SenseVoiceSmall/example/zh.mp3" --perform=True --loop=20
```
- 参数说明
- model_path: 模型权重路径
- om_model: om模型路径
- device: npu芯片id，默认使用0卡
- input: 输入mp3格式语音文件，这里以权重文件内的样例为例
- perform: 是否执行性能测试
- loop：性能测试，循环次数
  
推理执行完成后，会打屏语音文本的输出，和单次推理的耗时

### 3 性能
推理性能以'SenseVoiceSmall/example/zh.mp3'长度为8s的中文语音为例
|模型|芯片|端到端性能|
|------|------|------|
|sensevoice|800I A2|40ms|
|sensevoice|300I Pro|80ms|

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
