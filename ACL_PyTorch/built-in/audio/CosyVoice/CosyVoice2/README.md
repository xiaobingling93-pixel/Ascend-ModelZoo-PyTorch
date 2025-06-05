# CosyVoice-推理指导

- [CosyVoice-推理指导](#cosyvoice-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
    - [1 模型转换](#1-模型转换)
    - [2 开始推理验证](#2-开始推理验证)
    - [3 性能数据](#3-性能数据)

******

# 概述
‌Co‌syVoice是一款基于语音量化编码的语音生成大模型，能够深度融合文本理解和语音生成，实现自然流畅的语音体验。它通过离散化编码和依托大模型技术，能够精准解析并诠释各类文本内容，将其转化为宛如真人般的自然语音‌。CosyVoice2在原始1的基础上，把QWEN2模型接入CosyVoice的LLM部分，实现了推理加速

还可以通过参数 `--in_stream`设置模型为 **流式输入模式**，支持逐字符或分块输入文本，显著降低长文本生成延迟，提升实时交互体验。

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

## 获取本仓源码
```
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/CosyVoice/CosyVoice2
```

## 获取源码

1. 获取`PyTorch`源码
   ```
   # 获取CosyVoice源码
   git clone https://github.com/FunAudioLLM/CosyVoice
   cd CosyVoice
   git reset --hard fd45708
   git submodule update --init --recursive
   git apply ../diff_CosyVoice.patch
   # 获取Transformer源码
   git clone https://github.com/huggingface/transformers.git
   cd transformers
   git checkout v4.37.0
   cd ..
   # 将modeling_qwen模型文件替换到transformers仓内
   mv ../modeling_qwen2.py ./transformers/src/transformers/models/qwen2
   ```
   
    文件目录结构大致如下：
    ```text
    📁 CosyVoice/
    ├── 📁 CosyVoice2/
    |   |── 📄 diff_CosyVoice.patch
    |   |── 📄 modeling_qwen2.py
    |   |── 📁 CosyVoice
    |       |── 📁 cosyVoice源码文件    # cosyVoice的源码文件，此处不一一列举
    │       ├── 📁 CosyVoice-0.5B/     # 权重文件
    │       ├── 📁 transformers/   # transformers文件，里面有修改过的modeling_qwen2.py文件
    │       ├── 📄 infer.py        # 推理脚本
    │       └── 📄 modify_onnx.py  # 模型转换脚本
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

   本案例以CosyVoice2-0.5B为例，其他权重请自行适配。将下载下来的权重**放在CosyVoice目录下**。
   
    因cosyvoice2在2025年4月底更新过一次代码权重，因此下载时需要指定commit id，下载之前的权重。
    ```git
    # 1. 克隆
    git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git
    cd CosyVoice2-0.5B
    
    # 2. 切换到目标 commit
    git checkout 9bd5b08fc085bd93d3f8edb16b67295606290350
    
    # 3. 拉取 LFS 大文件（如模型权重）
    git lfs pull
    ```       

   本用例采用sft预训练音色推理，请额外下载spk权重放到权重目录下
   ```
   wget https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT/resolve/master/spk2info.pt
   ```

## 模型推理

### 1 模型转换

模型权重中提供了 `flow.decoder.estimator.fp32.onnx`和`speech_tokenizer_v2.onnx`两个onnx模型，对其进行结构修改后使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 修改onnx模型结构

   ```
   python3 modify_onnx.py ${CosyVoice2-0.5B}
   ```

   CosyVoice-300M是onnx模型所在权重文件夹，其他权重请自行更改权重文件名。执行该命令后会在CosyVoice-300M目录下生成修改后的onnx文件speech_token_md.onnx

2. 使用`ATC`工具将`ONNX`模型转为`OM`模型  

   配置环境变量

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   执行ATC命令，将利用npu-smi info命令获取的芯片型号填入${soc_version}中

   ```
   atc --framework=5 --soc_version=${soc_version} --model ./${CosyVoice2-0.5B}/speech_token_md.onnx --output ./${CosyVoice2-0.5B}/speech --input_shape="feats:1,128,-1;feats_length:1"
   atc --framework=5 --soc_version=${soc_version} --model ./${CosyVoice2-0.5B}/flow.decoder.estimator.fp32.onnx --output ./${CosyVoice2-0.5B}/flow --input_shape="x:2,80,-1;mask:2,1,-1;mu:2,80,-1;t:2;spks:2,80;cond:2,80,-1"
   atc --framework=5 --soc_version=${soc_version} --model ./${CosyVoice2-0.5B}/flow.decoder.estimator.fp32.onnx --output ./${CosyVoice2-0.5B}/flow_static --input_shape="x:2,80,-1;mask:2,1,-1;mu:2,80,-1;t:2;spks:2,80;cond:2,80,-1" --dynamic_dims="100,100,100,100;200,200,200,200;300,300,300,300;400,400,400,400;500,500,500,500;600,600,600,600;700,700,700,700" --input_format=ND
   ```
   在权重目录CosyVoice2-0.5B下会生成三个om模型, 分别为 speech_{arch}.om和flow_{arch}.om，flow_static.om。其中flow_static.om为分档模型，在流式输出中生效，档位设置为模型中默认流式输出token档位，如果在模型中修改token_hope_len，档位也需要对应修改。

   注：模型{arch}后缀为当前使用的CPU操作系统。

### 2 开始推理验证

   1. 首先移动infer.py文件到CosyVoice目录下


   2. 设置环境变量，执行推理命令

      ```
      # 1. 指定使用NPU ID，默认为0
      export ASCEND_RT_VISIBLE_DEVICES=0
      # 2. 设置环境变量
      export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
      export PYTHONPATH=transformers/src:$PYTHONPATH
      # 3. 执行推理脚本
      python3 infer.py --model_path=${CosyVoice2-0.5B} --stream_out
      ```
      - --model_path: 权重路径
      - --warm_up_times：warm up次数，默认为2
      - --infer_count：循环推理次数，默认为20
      - --stream_in：是否执行流式输入推理
      - --stream_out：是否执行流式输出推理

      在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，首次编译结束后，会在当前目录下生成.torchair_cache文件，后续推理无需重复编译，在warm_up结束后，会执行推理操作：
      * 非流式输入：将推理结果保存在`sft_i.wav`中，并打屏性能数据：实时率(rtf)，指的是平均1s时长的音频需要多少时间处理。
      * 流式输入：将推理结果保存在`stream_input_out_i.wav`文件中，并打屏性能数据：实时率(rtf)

### 3 性能数据

   | 模型        |芯片|rtf(实时率)|
   |-----------|------|------|
   | cosyvoice |800I A2|0.28s|

