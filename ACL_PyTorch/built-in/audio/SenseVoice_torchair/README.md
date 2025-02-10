# SenseVoice(TorchAir)-推理指导

- [SenseVoice(TorchAir)-推理指导](#sensevoicetorchair-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
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

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/FunAudioLLM/SenseVoice
   cd SenseVoice
   git reset --hard de00f2b
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```
   
3. 获取权重数据

   获取 https://modelscope.cn/models/iic/SenseVoiceSmall 文件夹，放在SenseVoice目录下

## 模型推理
执行patch文件，修改部分源码
```
cd ..
patch -p2 < ./diff_air.patch
```

移动推理py文件到源码目录内
```
mv infer_air.py ./SenseVoice
cd SenseVoice
```
执行推理命令

```
python3 infer_air.py --model_path=SenseVoiceSmall --device=0 --input="./SenseVoiceSmall/example/zh.mp3" --perform --loop=20
```
- 参数说明
- model_path: 模型权重路径
- device: npu芯片id，默认使用0卡
- input: 输入mp3格式语音文件，这里以权重文件内的样例为例
- perform: 是否执行性能测试
- loop：性能测试，循环次数
  
推理执行完成后，会打屏语音文本的输出，和单次推理的耗时

### 3 性能
推理性能以'SenseVoiceSmall/example/zh.mp3'长度为8s的中文语音为例
|模型|芯片|端到端性能|
|------|------|------|
|sensevoice|800I A2|49ms|
|sensevoice|300I Pro|92ms|

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
