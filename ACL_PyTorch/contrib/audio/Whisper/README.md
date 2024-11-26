# Whisper-推理指导



- [概述]
- [推理环境准备]
- [快速上手]
  - [获取源码]
  - [数据集准备]
  - [模型推理]
- [模型推理性能&精度]

******

# 概述

Whisper 是一个通用的语音识别模型。它在一个大型多样化音频数据集上进行训练。Whisper采用Transformer结构，是一个序列到系列的多任务模型，可用于各种语音处理任务，包括多语种语音识别、语音翻译、口语语言识别和语音活动检测，这些任务被表示为一个由解码器预测的token序列，使得单个模型就能完成传统多阶段的语音处理流水线，多任务训练格式使用一组特殊的token用作任务分类。

- 参考实现：

  ```
  url=https://github.com/openai/whisper
  ```
  
  
  
  ### 输入输出数据
  - encoder输入数据
  
    | 输入数据   | 数据类型  |      大小          | 数据排布格式 |
    |:-----:|:-------------------:|:------:|:----------:|
    | mel | FLOAT32 | batchsize x 80 x mel_len |  ND    |
  
  - encoder输出数据
  
    | 输出数据 |  数据类型   |         大小          | 数据排布格式 |
    |:-----------:|:-------------------:|:----------------:|:----------:|
    | n_layer_cross_k | FLOAT32   | batchsize x mel_len |   ND       |
    | n_layer_cross_v | FLOAT32   | 6 x batchsize x mel_len x 512| ND|
    
  - decoder输入数据
  
    | 输入数据   | 数据类型  |      大小          | 数据排布格式 |
    |:-----:|:-------------------:|:------:|:----------:|
    | tokens | INT64 | batchsize x n_tokens |  ND    |
    | in_n_layer_self_k_cache|FLOAT32|6 x batchsize x 448 x 512 | ND|
    | in_n_layer_self_v_cache|FLOAT32|6 x batchsize x 448 x 512 | ND|
    | n_layer_cross_k|FLOAT32|6 x batchsize x mel_len/2 x 512 | ND|
    | n_layer_cross_v|FLOAT32|6 x batchsize x mel_len/2 x 512 | ND|
    | offset|INT64|1 | ND|
    
  - decoder输出数据
  
    | 输出数据 |  数据类型   |         大小          | 数据排布格式 |
    |:-----------:|:-------------------:|:----------------:|:----------:|
    | logits | FLOAT32   | batchsize x mel_len |   ND       |
    | out_n_layer_self_k_cache | FLOAT32   | 6 x batchsize x mel_len x 512| ND|
    | out_n_layer_self_v_cache | FLOAT32   | 6 x batchsize x mel_len x 512| ND|
    
    


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 7.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.10.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手

## 获取源码

1. 获取本仓库`OM`推理代码
```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-Pytorch/ACL_Pytorch/contrib/audio/Whisper
```

2. 安装依赖

```shell
pip install -r requirements.txt
```

3. 安装ffmpeg

```shell
sudo apt-get install ffmpeg
```



## 数据准备

本模型使用一段音频文件作为输入，[测试数据地址](链接: https://pan.baidu.com/s/1xiHW7tmJe3lfAdQABWqsFA?pwd=gya6 提取码: gya6)如下，下载音频文件并存放在项目`data`目录下：

```	
Whisper_for_PyTorch
    ├── data      
    		└── test.wav
```



## 模型推理

### 1 模型转换

将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件

    下载[权重文件](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt)，以`base.en`为例，将下载的模型文件`base.en.pt`放在项目目录下。

2. 导出`ONNX`模型

   运行`pth2onnx.py`导出ONNX模型，并将原始模型中的配置信息与对应的`tokenizer`分别保存至`model_cfg.josn`和`tokens.txt `方便后续`om`模型推理时能读取对应的信息。由于whisper模型由`encoder`和`decoder`组成，且`encoder`和`decoder`需要进行`Cross Attention`操作，所以需要对模型进行修改，从而该脚本将会导出两个`ONNX `模型，即`encoder.onnx`和`decoder.onnx`。

   执行完这一步项目目录如下：

   ```
   Whisper_for_PyTorch
       ├── pth2onnx.py        
       ├── om_val.py             
       ├── encoder.onnx
       ├── decoder.onnx
       ├── model_cfg.json
       └── tokens.txt
   ```

3. 使用`ATC`工具将`ONNX`模型转为`OM`模型。

    1. 配置环境变量  

          ```shell
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          ```
        > **说明：**  
          该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   2. 执行命令查看芯片名称（得到`atc`命令参数中`soc_version`）

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

    3. 执行ATC命令  
    运行`atc.sh`导出`OM`模型。
        
        ```shell
        #运行脚本示例如下，可根据实际环境更改相应的参数
        bash atc.sh --enccoder_model=encoder --decoder_model=decoder --bs=1 --output_dir=output --soc=Ascend310P3
        ```
        
        ```shell
        #encoder模型实际执行的atc转换命令
        atc --framework=5 --input_format=ND --log=error --soc_version=${soc}
            --model=${encoder_model}.onnx --output=${output_dir}/${encoder_model}_bs${bs} 
            --input_shape="mel:${bs},80,250~3000"
        ```

        ```shell
        #decoder模型实际执行的atc转换命令
        atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
            --model=${decoder_model}.onnx --output=${output_dir}/${decoder_model}_bs${bs} \
            --input_shape="tokens:${bs},1~4;in_n_layer_self_k_cache:6,${bs},448,512; \
            in_n_layer_self_k_cache:6,${bs},448,512;n_layer_cross_k:6,${bs},250~3000,512;n_layer_cross_v:6,${bs},250~3000,512;offset:1"
        ```
        
        由于音频数据的长度不固定，再转`OM`模型时使用`~`来接受范围内的输入。除此之外，用户可以`padding`频谱长度，将动态值固定在一些档位，转而使用`--dynamic_dims`参数来接受动态输入。

        - 参数说明
          ：
          -   `--model`：ONNX模型文件
          -   `--framework`：5代表ONNX模型
          -   `--output`：输出的OM模型
          -   `--input_shape`：输入数据的shape
          -   `--log`：日志级别
          -   `--soc_version`：处理器型号

         运行成功后会生成`encoder_bs${bs}_{os}_{arch}.om`与`decoder_bs${bs}_{os}_{arch}.om`模型文件

### 2 开始推理验证

1. 安装`ais_bench`推理工具
   请访问[ais_bench](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，根据readme文件进行工具安装，建议使用whl包进行安装。

2. 执行推理 
   运行`om_val.py`推理`OM`模型，默认为转录任务，模型将输入的语音文件转化为对应的文字。若使用的为多语言模型则可进行翻译任务，将其他语种的语音文件，翻译为英文的语音文件。
   
   ```shell
   python3 om_val.py --encoder encoder_linux_x86_64.om \
                     --decoder decoder_linux_x86_64.om \
                     --tokens tokens.txt \
                     --model-cfg model_cfg.json \
                     data/test.wav
   ```
   
3. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证`OM`模型的性能，参考命令如下：
   
   ```
   python3 -m ais_bench --model encoder_bs${bs}_linux_x86_64.om --loop 10 --dymShape "mel:${bs},80,3000" --outputSize "10000000"
   ```
   ```
   python3 -m ais_bench --model decoder_bs${bs}_linux_x86_64.om --dymShape "tokens:${bs},1;in_n_layer_self_k_cache:6,${bs},448,512;in_n_layer_self_v_cache:6,${bs},448,512;n_layer_cross_k:6,${bs},351,512;n_layer_cross_v:6,${bs},351,512;offset:1" --outputSize "10000000,10000000,1000000"
   ```

   # 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
   | NPU芯片型号 | Batch Size |  mel_len|数据集  | throughout性能(encoder/decoder) |
   | :---------: | :--------: |:------:| :------: | :------------: |
   | 300I Pro |     1      |  2000  |  随机数据 |  76.60/56.58   |
   | 300I Pro |     1      |  1000  |  随机数据 |  184.52/57.11  |
   | 300I Pro |     2      |  2000  |  随机数据 |  76.58/78.70   |
   | 300I Pro |     2      |  1000  |  随机数据 |  203.29/80.05  |
   | 300I Pro |     4      |  2000  |  随机数据 |  74.73/125.31  |
   | 300I Pro |     4      |  1000  |  随机数据 |  194.89/130.61 |
   