# CosyVoice-推理指导

- [CosyVoice-推理指导](#cosyvoice-推理指导)
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
‌Co‌syvoice是一款基于语音量化编码的语音生成大模型，能够深度融合文本理解和语音生成，实现自然流畅的语音体验。它通过离散化编码和依托大模型技术，能够精准解析并诠释各类文本内容，将其转化为宛如真人般的自然语音‌

- 版本说明：
  ```
  url=https://github.com/FunAudioLLM/CosyVoice
  commit_id=bb690d
  model_name=Cosyvoice
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 24.0.RC3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  8.0.RC3 | -                                                                                                   |
  | Python                                                          |  3.8.20 | -                                                                                                     |
  | PyTorch                                                         | 2.4.0 | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.4.0 | -                                                                                                     |
  | 说明：Atlas 800I A2 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/FunAudioLLM/CosyVoice
   cd CosyVoice
   git reset --hard bb690d
   cd ..
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```
   
3. 安装msit工具
   
   参考[msit](https://gitee.com/ascend/msit)安装工具中的benchmark和surgen组件。
   

4. 获取权重数据

   从 https://modelscope.cn/models/iic/CosyVoice-300M 获取权重数据，放在当前目录下，使用当前目录下的cosyvoice.yaml，替换权重中的同名文件

## 模型推理

### 1 模型转换

模型权重中提供了 campplus.onnx，flow.decoder.estimator.fp32.onnx和speech_tokenizer_v1.onnx三个onnx模型，对其进行结构修改后使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 修改onnx模型结构
```
python3 modify_onnx.py ${CosyVoice-300M}
```
model_path是onnx模型所在权重文件夹，在当前目录下会生成campplus_md.onnx和speech_token_md.onnx文件

2. 简化onnx模型
```
onnxsim campplus_md.onnx campplus_sim.onnx
onnxsim speech_token_md.onnx speech_token_sim.onnx
onnxsim ./CosyVoice-300M/flow.decoder.estimator.fp32.onnx flow_sim.onnx
```
生成简化后的3个onnx模型

   
1. 使用`ATC`工具将`ONNX`模型转为`OM`模型  

配置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

执行ATC命令，以800I A2设备为例，利用npu-smi info命令获取芯片型号,填入soc_version参数中


```
atc --framework=5 --soc_version=${soc_version} --model campplus_sim.onnx --output campplus --input_shape="input:1,-1,80"
atc --framework=5 --soc_version=${soc_version} --model speech_token_sim.onnx --output speech --input_shape="feats:1,128,-1;feats_length:1"
atc --framework=5 --soc_version=${soc_version} --model flow_sim.onnx --output flow --input_shape="x:1,80,-1;mask:1,1,-1;mu:1,80,01;t:1;spks:1,80;cond:1,80,-1"
```
分别在当前目录生成3个OM模型

### 2 开始推理验证

修改源码以适配NPU推理
```
patch -p2 < diff.patch
```

移动推理py文件到源码目录内
```
mv infer.py ./CosyVoice/
```

进入源码目录下，安装第三方库
```
cd CosyVoice
git submodule update --init --recursive
```

设置环境变量，执行推理命令
```
export PYTHONPATH=third_party/Matcha-TTS
python3 infer.py --model_path=${CosyVoice-300M} --campplus=${campplus_om} --speech=${speech_om} --flow=${flow_om}
```
- --model_path: 权重路径
- --campplus：campplus的om模型文件
- --peech_token：peech_token的om模型文件
- --flow：flow的om模型文件

执行完成后，端到端平均推理耗时会打屏，生成的语言文件会保存在zero_shot.wav

### 3 性能
|模型|芯片|端到端性能|
|------|------|------|
|cosyvoice|800I A2|7.8s|

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
