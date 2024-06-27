# Zipformer流式模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理性能精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

（来自论文摘要）Conformer 已成为自动语音识别 (ASR) 中最流行的编码器模型。它将卷积模块添加到变压器中以学习局部和全局依赖性。
在这项工作中，我们描述了一种更快、内存效率更高、性能更好的转换器，称为 Zipformer。建模变化包括：1）类似 U-Net 的编码器结构，其中中间堆栈以较低的帧速率运行；
2）重新组织了具有更多模块的块结构，其中我们重新使用注意力权重以提高效率；3）LayerNorm的一种修改形式称为BiasNorm，允许我们保留一些长度信息；4）新的激活函数SwooshR和SwooshL比Swish效果更好。
我们还提出了一个新的优化器，称为 ScaledAdam，它通过每个张量的当前尺度来缩放更新以保持相对变化大致相同，并且还显式地学习参数尺度。它比 Adam 实现了更快的收敛和更好的性能。
在 LibriSpeech、Aishell-1 和 WenetSpeech 数据集上进行的大量实验证明了我们提出的 Zipformer 相对于其他最先进的 ASR 模型的有效性。
  

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                          | 版本          | 
|-----------------------------|-------------| 
| CANN                        | 8.0.RC2     | -                                                       |
| Python                      | 3.10.13     |                                                           
| torch                       | 2.1.0       |
| MindIE                      | 1.0.RC2.B071 |           
| NPU version                     | Ascend310P3 | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 环境安装

1. 安装k2
   1. （NPU）x86环境  
    ```shell
    wget https://huggingface.co/csukuangfj/k2/resolve/main/cpu/k2-1.24.4.dev20231220+cpu.torch2.0.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install k2-1.24.4.dev20231220+cpu.torch2.0.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    ```
    2. （NPU/GPU）arm环境，需要从源码编译。
    ```shell
    git clone https://github.com/k2-fsa/k2.git
    cd k2
    export K2_MAKE_ARGS="-j6"
    python3 setup.py install
    ```
    * **若编译失败，尝试再次编译前，需要先删除build文件夹。**
    * 若执行以上命令遇到错误，请参考[此链接](https://k2-fsa.github.io/k2/installation/from_source.html)。    
    3. （GPU）x86环境。从[此链接](https://k2-fsa.github.io/k2/cuda.html)下载对应CUDA版本的whl文件，然后使用pip进行安装。
    4. 验证k2是否安装成功  
    ```shell
    python3 -m k2.version
    ```
2. 安装其他依赖
    ```shell
    pip install lhotse
    pip install kaldifeat
    
    apt install libsndfile1
    ```
   * kaldifeat若安装失败，请执行以下命令使用源码安装：
    ```shell
    git clone https://github.com/csukuangfj/kaldifeat.git
    cd kaldifeat
    python3 setup.py install
    ```
3. 安装icefall
    ```shell
    git clone https://github.com/k2-fsa/icefall.git
   
    cd icefall
    git reset --hard e2fcb42f5f176d9e39eb38506ab99d0a3adaf202
    pip install -r requirements.txt
    ```
4. 将icefall加入环境变量, "/path/to/icefall"替换为icefall文件夹所在的路径。
   **这一步很重要，否则会报icefall找不到的错误。**
    ```shell
    export PYTHONPATH=/path/to/icefall:$PYTHONPATH
    ```

## 模型下载
从[此链接](https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615)下载模型相关文件。
模型转换和推理时只需要用到以下文件：  
    - data/lang_char/tokens.txt  
    - exp/epoch-12.pt

下载完后，整理成如下目录结构：
```shell
icefall-asr-zipformer-streaming-wenetspeech-20230615
├── data
│   └── lang_char
│       └── tokens.txt
└── exp
    └── epoch-12.pt
```

## 模型推理
1. 打代码补丁
    ```shell
    cd icefall
    
    cp egs/librispeech/ASR/zipformer/export-onnx-streaming.py egs/librispeech/ASR/zipformer/export-aie-streaming.py
    cp egs/librispeech/ASR/zipformer/onnx_pretrained-streaming.py egs/librispeech/ASR/zipformer/aie_pretrained-streaming.py
   
    patch -p1 < ../export_onnx.patch
    patch -p1 < ../export_aie.patch
    patch -p1 < ../aie_streaming_infer.diff
    ```
2. 将本代码仓的性能精度测试相关脚本（perf_test_aie.py, perf_test_onnx.py, precision_test.py, utils.py）
拷贝到icefall/egs/librispeech/ASR/zipformer目录下。

3. 导出onnx模型，用于精度测试。
   ```shell
   cd icefall/egs/librispeech/ASR/zipformer
   ```
   ```shell
   # 注意将`MODEL_PATH`修改为"icefall-asr-zipformer-streaming-wenetspeech-20230615"的绝对路径
   export MODEL_PATH=/absolute/path/to/icefall-asr-zipformer-streaming-wenetspeech-20230615
   ```
   ```shell
   python ./export-onnx-streaming.py \
     --tokens ${MODEL_PATH}/data/lang_char/tokens.txt \
     --use-averaged-model 0 \
     --epoch 12 \
     --avg 1 \
     --exp-dir ${MODEL_PATH}/exp \
     --num-encoder-layers "2,2,3,4,3,2" \
     --downsampling-factor "1,2,4,8,4,2" \
     --feedforward-dim "512,768,1024,1536,1024,768" \
     --num-heads "4,4,4,8,4,4" \
     --encoder-dim "192,256,384,512,384,256" \
     --query-head-dim 32 \
     --value-head-dim 12 \
     --pos-head-dim 4 \
     --pos-dim 48 \
     --encoder-unmasked-dim "192,192,256,256,256,192" \
     --cnn-module-kernel "31,31,15,15,15,31" \
     --decoder-dim 512 \
     --joiner-dim 512 \
     --causal True \
     --chunk-size 16 \
     --left-context-frames 128
    ```
   执行结束后，会在“icefall-asr-zipformer-streaming-wenetspeech-20230615/exp”目录下生成三个onnx文件：
    - encoder-epoch-12-avg-1-chunk-16-left-128.onnx
    - decoder-epoch-12-avg-1-chunk-16-left-128.onnx
    - joiner-epoch-12-avg-1-chunk-16-left-128.onnx
4. 导出torchscript模型，并进行编译。
   ```shell
   cd icefall/egs/librispeech/ASR/zipformer
   ```
   ```shell
   # 注意将`MODEL_PATH`修改为"icefall-asr-zipformer-streaming-wenetspeech-20230615"的绝对路径
   export MODEL_PATH=/absolute/path/to/icefall-asr-zipformer-streaming-wenetspeech-20230615
   ```
   ```shell
   # 注意将"icefall-asr-zipformer-streaming-wenetspeech-20230615"修改为实际路径
   python ./export-aie-streaming.py \
     --tokens ${MODEL_PATH}/data/lang_char/tokens.txt \
     --use-averaged-model 0 \
     --epoch 12 \
     --avg 1 \
     --exp-dir ${MODEL_PATH}/exp \
     --num-encoder-layers "2,2,3,4,3,2" \
     --downsampling-factor "1,2,4,8,4,2" \
     --feedforward-dim "512,768,1024,1536,1024,768" \
     --num-heads "4,4,4,8,4,4" \
     --encoder-dim "192,256,384,512,384,256" \
     --query-head-dim 32 \
     --value-head-dim 12 \
     --pos-head-dim 4 \
     --pos-dim 48 \
     --encoder-unmasked-dim "192,192,256,256,256,192" \
     --cnn-module-kernel "31,31,15,15,15,31" \
     --decoder-dim 512 \
     --joiner-dim 512 \
     --causal True \
     --chunk-size 16 \
     --left-context-frames 128
    ```
    执行结束后，会在“icefall-asr-zipformer-streaming-wenetspeech-20230615/exp”目录下生成三个编译好的torchscript文件：
    - encoder-epoch-12-avg-1-chunk-16-left-128_aie.pt
    - decoder-epoch-12-avg-1-chunk-16-left-128_aie.pt
    - joiner-epoch-12-avg-1-chunk-16-left-128_aie.pt  
    当前目录下还会生成两个json文件，用于后续推理时加载模型：
    - encoder_meta_data.json
    - decoder_meta_data.json
5. 运行推理样例
   1. 下载样例语音数据
      ```shell
      cd icefall/egs/librispeech/ASR/zipformer
      wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
      ```
   2. 执行推理
      ```shell
      # 注意将"icefall-asr-zipformer-streaming-wenetspeech-20230615"修改为实际路径
      python ./aie_pretrained-streaming.py \
        --encoder-meta-data-path=./encoder_meta_data.json \
        --decoder-meta-data-path=./decoder_meta_data.json \
        --encoder-model-filename ${MODEL_PATH}/exp/encoder-epoch-12-avg-1-chunk-16-left-128_aie.pt \
        --decoder-model-filename ${MODEL_PATH}/exp/decoder-epoch-12-avg-1-chunk-16-left-128_aie.pt \
        --joiner-model-filename ${MODEL_PATH}/exp/joiner-epoch-12-avg-1-chunk-16-left-128_aie.pt \
        --tokens ${MODEL_PATH}/data/lang_char/tokens.txt \
        ./zh.wav
      ```
      执行结束后，会在命令行看到如下输出，说明推理成功，且结果正确：
      ```shell
      INFO [aie_pretrained-streaming.py:554] 我认为跑步最重要的就是给我带来了身体健康
      ```
      
6. 精度测试
   ```shell
   cd icefall/egs/librispeech/ASR/zipformer
   
    # 注意将"icefall-asr-zipformer-streaming-wenetspeech-20230615"修改为实际路径
   python precision_test.py \
     --encoder_onnx_path=${MODEL_PATH}/exp/encoder-epoch-12-avg-1-chunk-16-left-128.onnx \
     --encoder_aie_path=${MODEL_PATH}/exp/encoder-epoch-12-avg-1-chunk-16-left-128_aie.pt \
     --decoder_onnx_path=${MODEL_PATH}/exp/decoder-epoch-12-avg-1-chunk-16-left-128.onnx \
     --decoder_aie_path=${MODEL_PATH}/exp/decoder-epoch-12-avg-1-chunk-16-left-128_aie.pt \
     --joiner_onnx_path=${MODEL_PATH}/exp/joiner-epoch-12-avg-1-chunk-16-left-128.onnx \
     --joiner_aie_path=${MODEL_PATH}/exp/joiner-epoch-12-avg-1-chunk-16-left-128_aie.pt
   ```
    执行结束后，会在命令行看到如下输出，说明三个模型的精度均达标，即MindIE ts模型与onnx模型每个输出节点的相似度均大于0.99：
    ```shell
   === Compare the outputs of ONNX and AIE ===
   Start comparing encoder...
   Number of outputs to compare: 99
   Number of outputs with cosine similarity > 0.99: 99
   
   Start comparing decoder...
   Number of outputs to compare: 1
   Number of outputs with cosine similarity > 0.99: 1
   
   Start comparing joiner...
   Number of outputs to compare: 1
   Number of outputs with cosine similarity > 0.99: 1
    ```
7. 性能测试
   1. MindIE ts模型性能测试
      ```shell
      cd icefall/egs/librispeech/ASR/zipformer
      
      python perf_test_aie.py \
        --encoder_meta_data_path=./encoder_meta_data.json \
        --encoder_aie_path=${MODEL_PATH}/exp/encoder-epoch-12-avg-1-chunk-16-left-128_aie.pt \
        --decoder_aie_path=${MODEL_PATH}/exp/decoder-epoch-12-avg-1-chunk-16-left-128_aie.pt \
        --joiner_aie_path=${MODEL_PATH}/exp/joiner-epoch-12-avg-1-chunk-16-left-128_aie.pt \
        --device_id=0
      ```
      执行结束后，三个模型的性能信息会打印在命令行，如下所示：
      ```shell
      Encoder latency: 19.92 ms
      Encoder throughput: 50.19 fps
      Decoder latency: 0.19 ms
      Decoder throughput: 5353.36 fps
      Joiner latency: 0.23 ms
      Joiner throughput: 4387.90 fps
      ```
   2. onnx模型性能测试。  
      1. （可选）若使用GPU，请确保已安装CUDA和pytorch-gpu版本，同时需安装onnxruntime-gpu，如下所示：
      ```shell
      pip uninstall onnxruntime
      pip install onnxruntime-gpu
      ```
      验证onnxruntime-gpu是否安装成功：
      ```python
      import onnxruntime
      print(onnxruntime.get_device())  # 若输出为GPU，则说明安装成功
      ``` 
      2. 执行性能测试。  
      ```shell
      cd icefall/egs/librispeech/ASR/zipformer
      
      python perf_test_onnx.py \
        --encoder_path ${MODEL_PATH}/exp/encoder-epoch-12-avg-1-chunk-16-left-128.onnx \
        --decoder_path ${MODEL_PATH}/exp/decoder-epoch-12-avg-1-chunk-16-left-128.onnx \
        --joiner_path ${MODEL_PATH}/exp/joiner-epoch-12-avg-1-chunk-16-left-128.onnx \
        --use_gpu  # 若使用CPU，请删除此参数
      ```
      执行结束后，三个模型的性能信息会打印在命令行，如下所示：
      ```shell
      Encoder latency: 58.07 ms
      Encoder throughput: 17.22 fps
      Decoder latency: 1.80 ms
      Decoder throughput: 555.71 fps
      Joiner latency: 0.20 ms
      Joiner throughput: 5032.52 fps
      ```

# 模型推理性能精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

Zipformer流式模型由三个子模型组成，分别是encoder、decoder和joiner，其性能如下表所示：

| 模型      | pt插件 - 310P性能（时延/吞吐率） | T4性能（时延/吞吐率）       | A10性能（时延/吞吐率）      |
|---------|-----------------------|--------------------|--------------------|
| encoder | 20.4 ms / 49 fps      | 24.7 ms / 40 fps   | 19 ms / 52 fps     |
| decoder | 0.19 ms / 5156 fps    | 0.59 ms / 1684 fps | 0.13 ms / 7604 fps |
| joiner  | 0.22 ms / 4448 fps    | 0.13 ms / 7645 fps | 0.11 ms / 9224 fps |
| 端到端     | 20.81 ms / 48 fps     | 25.42 ms /  39 fps | 19.24 ms / 52 fps  |

