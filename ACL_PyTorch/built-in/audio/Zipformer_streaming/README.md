# Zipformer流式模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理性能精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

（来自论文摘要）Conformer 已成为自动语音识别 (ASR) 中最流行的编码器模型。它将卷积模块添加到变压器中以学习局部和全局依赖性。 在这项工作中，我们描述了一种更快、内存效率更高、性能更好的转换器，称为 Zipformer。建模变化包括：1）类似 U-Net 的编码器结构，其中中间堆栈以较低的帧速率运行； 2）重新组织了具有更多模块的块结构，其中我们重新使用注意力权重以提高效率；3）LayerNorm的一种修改形式称为BiasNorm，允许我们保留一些长度信息；4）新的激活函数SwooshR和SwooshL比Swish效果更好。 我们还提出了一个新的优化器，称为 ScaledAdam，它通过每个张量的当前尺度来缩放更新以保持相对变化大致相同，并且还显式地学习参数尺度。它比 Adam 实现了更快的收敛和更好的性能。 在 LibriSpeech、Aishell-1 和 WenetSpeech 数据集上进行的大量实验证明了我们提出的 Zipformer 相对于其他最先进的 ASR 模型的有效性。
  

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

  | 配套                                                         | 版本      |
  |---------|---------|
  | 固件与驱动                                                   | 23.0.rc1  | 
  | CANN                                                         | 7.0.RC1 |
  | Python                                                       | 3.9.11  |
  | PyTorch                                                      | 2.0.1   |

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
    若执行以上命令遇到错误，请参考[此链接](https://k2-fsa.github.io/k2/installation/from_source.html)。
    3. (GPU) x86环境。从[此链接](https://k2-fsa.github.io/k2/cuda.html)下载对应CUDA版本的whl文件，然后使用pip进行安装。
    4. 验证k2是否安装成功  
    ```shell
    python3 -m k2.version
    ```
2. 安装其他依赖
    ```shell
    pip install lhotse
    pip install kaldifeat
    pip install onnxsim
    ```
3. 安装icefall
    ```shell
    git clone https://github.com/k2-fsa/icefall.git
    git reset --hard e2fcb42f5f176d9e39eb38506ab99d0a3adaf202
   
    cd icefall
    pip install -r requirements.txt
    ```
4. 安装onnx改图工具
    ```shell
    git clone https://gitee.com/ascend/msadvisor.git
    cd msadvisor/auto-optimizer
    python3 -m pip install .
    ```
5. 将icefall加入环境变量, "/path/to/icefall"替换为icefall文件夹所在的路径。
   **这一步很重要，否则会报icefall找不到的错误。**
    ```shell
    export PYTHONPATH=/path/to/icefall:$PYTHONPATH
    ```

## 模型下载
1. 安装 git lfs
    ```shell
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

    sudo apt-get install git-lfs
    git lfs install --skip-repo
    ```
2. 下载模型
    ```shell
   git clone https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
    ```
   若下载失败，请尝试从以上链接手动下载文件。模型转换和推理时只需要用到以下文件：
    - data/lang_char/tokens.txt
    - exp/epoch-12.pt

## 模型转换
1. 打代码补丁
    ```shell
    cd icefall  # 在此目录下使用本代码仓的补丁

    # 保留onnx推理代码的备份
    cp egs/librispeech/ASR/zipformer/onnx_pretrained-streaming.py egs/librispeech/ASR/zipformer/om_pretrained-streaming.py

    patch -p1 < om_infer.diff
    ```

2. 导出onnx模型
    ```shell
   cd icefall/egs/librispeech/ASR/zipformer
   # 注意将"icefall-asr-zipformer-streaming-wenetspeech-20230615"修改为实际路径
   python ./export-onnx-streaming.py \
     --tokens icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
     --use-averaged-model 0 \
     --epoch 12 \
     --avg 1 \
     --exp-dir icefall-asr-zipformer-streaming-wenetspeech-20230615/exp \
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

3. 使用onnxsim进行常量折叠
   ```shell
   cd icefall-asr-zipformer-streaming-wenetspeech-20230615/exp
   
   onnxsim encoder-epoch-12-avg-1-chunk-16-left-128.onnx encoder-epoch-12-avg-1-chunk-16-left-128_sim.onnx
   onnxsim decoder-epoch-12-avg-1-chunk-16-left-128.onnx decoder-epoch-12-avg-1-chunk-16-left-128_sim.onnx
   ```
   
4. 使用本代码仓的 `modify_decoder.py`进行改图，以便正确导出om
    ```shell
    python modify_decoder.py --onnx icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-1-chunk-16-left-128_sim.onnx
    ```
    执行结束后，会在`icefall-asr-zipformer-streaming-wenetspeech-20230615/exp`目录下生成`decoder-epoch-12-avg-1-chunk-16-left-128_sim_modified.onnx`文件。

5. 将本代码仓的`atc.sh`拷贝到`icefall-asr-zipformer-streaming-wenetspeech-20230615/exp`目录下，执行以下命令导出om  
    ```shell
    sh atc.sh
    ```
    执行结束后，会在当前目录下生成三个om文件：
    - encoder.om
    - decoder.om
    - joiner.om

## 模型推理
1. 运行推理样例
   1. 下载样例语音数据
      ```shell
      cd icefall/egs/librispeech/ASR/zipformer
      wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
      ```
   2. 执行推理
      ```shell
      # 注意将"icefall-asr-zipformer-streaming-wenetspeech-20230615"修改为实际路径
      python ./om_pretrained-streaming.py \
      --encoder-meta-data-path=./encoder_meta_data.json \
      --decoder-meta-data-path=./decoder_meta_data.json \
      --encoder-model-filename icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder.om \
      --decoder-model-filename icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder.om \
      --joiner-model-filename icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner.om \
      --tokens icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
      ./zh.wav
      ```
      执行结束后，会在命令行看到如下输出，说明推理成功，且结果正确：
      ```shell
      我认为跑步最重要的就是给我带来了身体健康
      ```

2. 将本代码仓的性能精度测试相关脚本（perf_test_om.py, perf_test_onnx.py, precision_test.py, utils.py）拷贝到`icefall/egs/librispeech/ASR/zipformer`目录下。
      
6. 精度测试。
   ```shell
   cd icefall/egs/librispeech/ASR/zipformer
   
   # 注意将"icefall-asr-zipformer-streaming-wenetspeech-20230615"修改为实际路径
   python precision_test.py \
    --encoder_onnx_path=icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-1-chunk-16-left-128_sim.onnx \
    --encoder_om_path=icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder_linux_aarch64.om \
    --decoder_onnx_path=icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-1-chunk-16-left-128_sim_modified.onnx \
    --decoder_om_path=icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder.om \
    --joiner_onnx_path=icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-1-chunk-16-left-128.onnx \
    --joiner_om_path=icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner.om
   ```
    执行结束后，会在命令行看到如下输出，说明三个模型的精度均达标，即om与onnx模型每个输出节点的相似度均大于0.99：
    ```shell
   === Compare the outputs of ONNX and OM ===
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
   1. om模型性能测试
      ```shell
      # 注意将"icefall-asr-zipformer-streaming-wenetspeech-20230615"修改为实际路径
      python -m ais_bench --model icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder.om 

      python -m ais_bench --model icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder.om

      python -m ais_bench --model icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner.om
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
        --encoder_path icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-1-chunk-16-left-128.onnx \
        --decoder_path icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-1-chunk-16-left-128.onnx \
        --joiner_path icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-1-chunk-16-left-128.onnx \
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

| 模型      | om - 310P性能（时延/吞吐率） | T4性能（时延/吞吐率）       | A10性能（时延/吞吐率）      |
|---------|-----------------------|--------------------|--------------------|
| encoder | 12.7 ms / 79 fps      | 24.7 ms / 40 fps   | 19 ms / 52 fps     |
| decoder | 0.13 ms / 7604 fps    | 0.59 ms / 1684 fps | 0.13 ms / 7604 fps |
| joiner  | 0.13 ms / 7604 fps    | 0.13 ms / 7645 fps | 0.11 ms / 9224 fps |
| 端到端  | 12.96 ms / 77 fps    | 25.42 ms / 39 fps | 19.24 ms / 52 fps |
