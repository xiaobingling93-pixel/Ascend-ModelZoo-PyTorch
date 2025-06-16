# CosyVoice(TorchAir)-推理指导

- [CosyVoice-推理指导](#cosyvoice(TorchAir)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
    - [1 模型转换](#1-模型转换)
    - [2 开始推理验证](#2-开始推理验证)
    - [3 性能](#3-性能)

******

# 概述
&emsp;&emsp;‌Co‌syVoice是一款基于语音量化编码的语音生成大模型，能够深度融合文本理解和语音生成，实现自然流畅的语音体验。它通过离散化编码和依托大模型技术，能够精准解析并诠释各类文本内容，将其转化为宛如真人般的自然语音‌

- 版本说明：
  ```
  url=https://github.com/FunAudioLLM/CosyVoice
  commit_id=fd45708
  model_name=Cosyvoice
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 24.0.RC3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  8.0.RC3 | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                          |  3.8 | -                                                                                                     |
  | PyTorch                                                         | 2.4.0 | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.4.0.post2 | -                                                                                                     |
  | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/CosyVoice
   ```

1. 获取`PyTorch`源码
   ```
   git clone https://github.com/FunAudioLLM/CosyVoice
   cd CosyVoice
   git reset --hard fd45708
   git submodule update --init --recursive
   # 如果使用800I系列推理卡，使用diff_800I.patch, 300I系列对应diff_300I.patch
   git apply ../diff_{type}.patch
   ```
   
2. 安装依赖  
   ```
   pip3 install -r ../requirements.txt
   apt-get install sox # centos版本 yum install sox
   ```
   注：如果遇到无法安装WeTextProcessing的场景，可以参考以下方法手动安装编译
   ```bash
   # 下载安装包并解压
   wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz
   # 进入目录后编译安装
   ./configure --enable-far --enable-mpdt --enable-pdt
   make -j$(nproc)
   make install
   # 确认动态库文件存在：
   ls /usr/local/lib/libfstmpdtscript.so.26
   # 配置动态库路径
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   sudo ldconfig
   # 安装WeTextProcessing
   pip3 install WeTextProcessing==1.0.4.1
   ```
   

3. 安装msit工具
   
   参考[msit](https://gitee.com/ascend/msit)安装工具中的benchmark和surgen组件。（未安装会提示 ais_bench 导入失败报错）
   

4. 获取权重数据

   本案例以CosyVoice-300M为例，其他权重请自行适配

   获取 https://modelscope.cn/models/iic/CosyVoice-300M 权重文件夹，放在CosyVoice目录下

   或者通过git方式获取
   ```
   # git模型下载，请确保已安装git lfs
   git clone https://www.modelscope.cn/iic/CosyVoice-300M.git CosyVoice/CosyVoice-300M
   ```

5. 文件结构如下：
    ```text
    📁 CosyVoice/
    ├── 📁 CosyVoice/
    |   |── 📁 CosyVoice的源码文件    # CosyVoice其他的源码文件，此处不一一列举
    │   ├── 📁 CosyVoice-300M/  # 权重文件
    │   ├── 📄 infer.py        # 推理脚本
    │   └── 📄 modify_onnx.py  # 模型转换脚本
    ├── 📄 diff_300I.patch
    └── 📄 diff_800I.patch
    ```


## 模型推理

### 1 模型转换

&emsp;&emsp;&emsp;模型权重中提供了 flow.decoder.estimator.fp32.onnx和speech_tokenizer_v1.onnx两个onnx模型，对其进行结构修改后使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 修改onnx模型结构

   ```
   python3 modify_onnx.py ${CosyVoice-300M}
   ```

   CosyVoice-300M是onnx模型所在权重文件夹，其他权重请自行更改权重文件名。执行该命令后会在CosyVoice-300M目录下生成修改后的onnx文件speech_token_md.onnx

2. 使用`ATC`工具将`ONNX`模型转为`OM`模型  

   配置环境变量

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   执行ATC命令，将利用npu-smi info命令获取的芯片型号填入${soc_version}中

   ```
   atc --framework=5 --soc_version=${soc_version} --model ./${CosyVoice-300M}/speech_token_md.onnx --output ./${CosyVoice-300M}/speech --input_shape="feats:1,128,-1;feats_length:1"
   atc --framework=5 --soc_version=${soc_version} --model ./${CosyVoice-300M}/flow.decoder.estimator.fp32.onnx --output ./${CosyVoice-300M}/flow --input_shape="x:2,80,-1;mask:2,1,-1;mu:2,80,-1;t:2;spks:2,80;cond:2,80,-1"
   ```
   在权重目录CosyVoice-300M下会生成两个om模型, 分别为 speech_{arch}.om和flow_{arch}.om

   注：模型{arch}后缀为当前使用的CPU操作系统。

### 2 开始推理验证

   1. **首先移动infer.py文件到CosyVoice目录下**


   2. 设置环境变量，执行推理命令

      ```
      # 1. 指定使用NPU ID，默认为0
      export ASCEND_RT_VISIBLE_DEVICES=0
      # 2. 设置环境变量
      export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
      # 3. 执行推理脚本
      python3 infer.py --model_path=${CosyVoice-300M} 
      ```
      - --model_path: 权重路径
      - --warm_up_times：warm up次数，默认为2
      - --infer_count：循环推理次数，默认为10

      在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并将推理结果保存在'zero_shot_result.wav'中，并打屏性能数据：实时率(rtf)，指的是平均1s时长的音频需要多少时间处理。

### 3 性能

   |模型|芯片|rtf(实时率)|
   |------|------|------|
   |cosyvoice|800I A2|0.7s|
   |cosyvoice|300I DUO(单芯)|2.0s|

