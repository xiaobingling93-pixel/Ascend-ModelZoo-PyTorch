# Whisper-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理性能精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

([来自开源代码仓](https://github.com/openai/whisper))Whisper是一种通用语音识别模型。它是在各种音频的大型数据集上训练的，也是一个多任务模型，可以执行多语言语音识别、语音翻译和语言识别。
  

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                          | 版本          | 
|-----------------------------|-------------| 
| CANN                        | 8.0.RC1     | -                                                       |
| Python                      | 3.10.13     |                                                           
| torch                       | 2.1.0       |
| Ascend-mindie-rt_1.0.RC1    | -           
| Ascend-mindie-torch-1.0.RC1 | -           
| 芯片类型                        | Ascend310P3 | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 源码下载
    ```
    git clone https://github.com/openai/whisper.git
    cd whisper
    git reset --hard ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab
    ```
2. 模型导出
    ```
    patch -p1 < ../trace_model.patch
    pip3 install .
    cd ..
    wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
    mkdir /tmp/models
    whisper zh.wav --model tiny
    ```
    完成该步骤将在`/tmp/models`目录下生成`encoder.ts`, `decoder_prefill.ts`, `decoder_decode.ts`3个文件。
    注：如需修改模型路径，可在打完补丁后手动修改`whisper/decoding.py`和`whisper/model.py`文件，后续步骤模型推理同样需要修改对应模型的载入路径。

3. 模型编译
    ```
    python3 compile.py
    ```
    请忽略命令行的报错信息，执行完成后将在`/tmp/models`目录下生成`encoder_compiled.ts`, `language_detection_compiled.ts`, `decoder_prefill_compiled.ts`, `decoder_decode_compiled.ts`四个文件。
    
    参数说明：
    - --model_path：导出的Torchscript模型路径，模型编译后保存在同一路径， 默认为`/tmp/models`。
    - --beam_size: 集束搜索参数，默认为5。与推理参数保持一致，如模型导出时指定了该参数，在编译时需要保持一致。
    - --nblocks: 模型Blocks参数，跟模型大小相关，tiny 4, base 6, small 12, medium 24, large 32。
    - --soc_version: 芯片类型，当前仅在Ascend310P3上调试。

4. 模型推理
    ```
    cd whisper
    git reset --hard ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab
    patch -p1 < ../torch_aie_infer.patch
    pip3 install .
    cd ..
    whisper zh.wav --model tiny
    ```
    推理结束后，会在命令行打印出如下输出：
    ```
    [00:00.000 --> 00:04.480] 我認為跑步最重要的就是給我帶來了身體健康
    ```
    如需要简体输出，可使用如下命令：
    ```
    whisper zh.wav --model tiny --initial_prompt "简体翻译："
    ```

    注：默认`芯片ID为0`，模型路径为`/tmp/models`。如需修改，可在打完补丁后手动修改`whisper/decoding.py`和`whisper/model.py`文件，可使用全局替换文件中的`npu:0`, `/tmp/models`, `torch_aie.set_device(0)`。


# 模型推理性能精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

待后续补充。
