# Whisper-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理性能精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

([来自开源代码仓](https://github.com/openai/whisper))Whisper是一种通用语音识别模型。它是在各种音频的大型数据集上训练的，也是一个多任务模型，可以执行多语言语音识别、语音翻译和语言识别。
  

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表
  | 配套    | 版本     |
  |---------| ------- |
  | 固件与驱动 | 24.1.rc1  |
  | CANN | 8.0.rc1 |
  | Python | 3.10.13 |
  | PyTorch | 2.1.0 |
  | Ascend-mindie-rt1.0.RC1 | - |
  | Ascend-mindie-torch-1.0.RC1 | - |

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
    执行上述步骤需要依赖`ffmpeg`，ubuntu下可通过`apt-get install ffmpeg`安装。完成上述步骤将在`/tmp/models`目录下生成`encoder.ts/onnx`, `decoder_prefill.ts/onnx`, `decoder_decode.onnx`6个文件。
    注：如需修改模型路径，可在打完补丁后手动修改`whisper/decoding.py`和`whisper/model.py`文件，后续步骤模型推理同样需要修改对应模型的载入路径。

3. 模型编译
    ```
    python3 compile.py
    ```
    执行完成后将在`/tmp/models`目录下生成`encoder_compiled.ts`, `language_detection_compiled.ts`, `decoder_prefill_compiled.ts`, `decoder_decode_compiled.ts`四个文件。
    
    参数说明：
    - --model_path：导出的Torchscript模型路径，模型编译后保存在同一路径， 默认为`/tmp/models`。
    - --beam_size: 集束搜索参数，默认为5。与推理参数保持一致，如模型导出时指定了该参数，在编译时需要保持一致。
    - --nblocks: 模型Blocks参数，跟模型大小相关，tiny 4, base 6, small 12, medium 24, large 32。
    - --soc_version: 芯片类型，当前仅在Ascend310P3上调试。

4. 模型推理
    ```
    cd whisper
    git reset --hard ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab
    patch -p1 < ../mindietorch_infer.patch
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

1. 精度验证
    ```
    python3 precision_test.py
    ```
    
    参数说明：
    - --sim_threshold: 余弦相似度阈值，默认0.99。
    - --ntokens: prefill阶段输入token数量，decode阶段缓存token数量，默认100。

    执行结束后，期望输出如下：
    ```
    === Compare the outputs of ONNX and AIE ===
    Start comparing encoder...
    Number of outputs to compare: 1
    Number of outputs with cosine similarity > 0.99: 1
    Number of outputs to compare: 3
    Number of outputs with cosine similarity > 0.99: 3
    Number of outputs to compare: 3
    Number of outputs with cosine similarity > 0.99: 3
    ```

2. 性能验证

    a) aie模型性能测试
    ```
    python perf_test_aie.py
    ```

    执行结束后，期望输出如下：
    ```
    Encoder latency: 7.75 ms
    Encoder throughput: 128.97 fps
    Decoder prefill latency: 10.14 ms
    Decoder prefill throughput: 98.63 fps
    Decoder decode latency: 2.92 ms
    Decoder decode throughput: 342.55 fps
    ```

    b) onnx模型性能测试
    （可选）若使用GPU，请确保已安装CUDA和pytorch-gpu版本，同时需安装onnxruntime-gpu，如下所示：
    ```shell
    pip uninstall onnxruntime
    pip install onnxruntime-gpu
    ```
    验证onnxruntime-gpu是否安装成功：
    ```python
    import onnxruntime
    print(onnxruntime.get_device())  # 若输出为GPU，则说明安装成功
    ``` 
    执行性能测试
    ```
    python perf_test_onnx.py --use_gpu
    ```

    参数说明：
    - --use_gpu: 使能gpu推理，不加该选项默认cpu。

    执行结束后，期望输出如下：
    ```
    Encoder latency: 59.49 ms
    Encoder throughput: 16.81 fps
    Decoder prefill latency: 141.14 ms
    Decoder prefill throughput: 7.09 fps
    Decoder decode latency: 36.05 ms
    Decoder decode throughput: 27.74 fps
    ```

    
    | 模型    | pt插件 - 310P性能（时延/吞吐率） | T4性能（时延/吞吐率） | A10性能（时延/吞吐率）|
    |---------|--------------------------------|---------------------|--------------------|
    | encoder | 7.75 ms / 128.97 fps | 9.31 ms / 107.47 fps | 4.21 ms / 237.50 fps |
    | prefill | 10.14 ms / 98.63 fps | 72.08 ms / 13.87 fps | 45.15 ms / 22.15 fps |
    | decode  | 2.92 ms / 342.55 fps | 10.46 ms / 95.62 fps | 4.91 ms / 203.61 fps |

    注：在实际推理中encoder和prefill均调用一次，decode会调用多次（上面数据假设缓存token长度为100）。并且在whisper全流程推理中还包括后处理，cache重新排布等步骤，以上数据仅作参考。