# Conformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理性能精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Conformer模型是一种混合神经网络架构，专门设计用于处理序列到序列的任务，如自动语音识别（ASR）。它融合了卷积神经网络（CNN）和自注意力机制（来自Transformer模型）的优点，旨在捕捉序列数据的局部特征和全局依赖。Conformer通过在其架构中巧妙地结合这两种方法，有效地处理了时间序列数据的复杂性，比如语音波形，从而在许多任务上实现了卓越的性能。简而言之，Conformer通过集成CNN的强大特征提取能力和Transformer的高效序列建模能力，为序列分析任务提供了一种强大的解决方案。


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

  | 配套                                                         | 版本      |
  |---------|---------|
  | CANN                        | 8.0RC1     |
  | Python                      | 3.10.13     |
  | torch                       | 2.1.0       |
  | 芯片类型                        | Ascend310P3 |

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
    export K2_MAKE_ARGS="-j"
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
    ```

    如果安装kaldifeat失败，请参考以下命令进行源码安装

    ```sh
    git clone https://github.com/csukuangfj/kaldifeat.git
    cd kaldifeat
    python3 setup.py install
    ```

3. 安装icefall

    ```shell
    git clone https://github.com/k2-fsa/icefall.git
    git reset --hard e2fcb42f5f176d9e39eb38506ab99d0a3adaf202
       
    cd icefall
    pip install -r requirements.txt
    ```

4. 将icefall加入环境变量, "/path/to/icefall"替换为icefall文件夹所在的路径。
   **这一步很重要，否则会报icefall找不到的错误。**
   
    ```shell
    export PYTHONPATH=/path/to/icefall:$PYTHONPATH
    ```

## 模型转换
1. 下载模型

   从以下HuggingFace链接下载所需文件，所需文件为`./exp/pretrained_epoch_9_avg_1.pt`, 和`./data`整个文件夹
   https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_offline/tree/main
   
   进入上一步中的`./icefall`路径，并`cd`到`./egs/wenetspeech/ASR/`中，新建`exp`目录。
   
   将整个`data`文件夹复制到`./ASR`中。
   
   将`./pretrained_epoch_9_avg_1.pt`复制到`./ASR/exp`目录下，并**重命名为epoch-99.pt**。
   
2. 复制代码文件

   将如下代码文件使用`cp`命令复制到`egs/wenetspeech/ASR`下：

   ```
   conformer_py.patch
   export_torchscript.patch
   perf_test_aie.py
   precision_test_onnx.py
   ```

   将如下代码文件使用`cp`命令复制到`egs/wenetspeech/ASR/pruned_transducer_stateless5`下：

   ```
   encoder_compile.py
   decoder_compile.py
   joiner_compile.py
   ```

前两个步骤最终达到的效果如下所示：

```
egs/wenetspeech/ASR/
├── data
│   └── lang_char
│       └── Linv.pt等其他文件
|
├── exp
│   └── epoch-99.pt
|
├── conformer_py.patch
├── export_torchscript.patch
|
├── pruned_transducer_stateless5
|   ├── encoder_compile.py
|   ├── decoder_compile.py
|   └── joiner_compile.py
|
├── perf_test_aie.py
└── precision_test_onnx.py
```

3. 导出ONNX模型

   > *注意，如果曾经修改过conformer.py，需要还原此文件* 
   >
   > git checkout -- ./pruned_transducer_stateless5/conformer.py

   在`egs/wenetspeech/ASR`路径下执行如下命令：

   ```sh
   python3 ./pruned_transducer_stateless5/export-onnx.py \
     --tokens ./data/lang_char/tokens.txt \
     --epoch 99 \
     --avg 1 \
     --use-averaged-model 0 \
     --exp-dir ./exp \
     --num-encoder-layers 24 \
     --dim-feedforward 1536 \
     --nhead 8 \
     --encoder-dim 384 \
     --decoder-dim 512 \
     --joiner-dim 512
   ```

   生成的ONNX模型文件会出现在`./exp`目录下。

4. 导出ts模型

   1. 修改原始模型

      在`egs/wenetspeech/ASR`路径下执行如下命令：

      ```sh
      #先应用patch文件
      patch ./pruned_transducer_stateless5/conformer.py conformer_py.patch
      patch --force ./pruned_transducer_stateless5/export-onnx.py ./export_torchscript.patch -o ./pruned_transducer_stateless5/export_torchscript.py
      #再导出torchscript模型
      python3 ./pruned_transducer_stateless5/export_torchscript.py \
        --tokens ./data/lang_char/tokens.txt \
        --epoch 99 \
        --avg 1 \
        --use-averaged-model 0 \
        --exp-dir ./exp \
        --num-encoder-layers 24 \
        --dim-feedforward 1536 \
        --nhead 8 \
        --encoder-dim 384 \
        --decoder-dim 512 \
        --joiner-dim 512
      ```

   2. 转为MindIETorch模型

      在`egs/wenetspeech/ASR`路径下执行如下命令：

      ```sh
      #在ASR目录下执行
      python ./pruned_transducer_stateless5/encoder_compile.py
      python ./pruned_transducer_stateless5/decoder_compile.py
      python ./pruned_transducer_stateless5/joiner_compile.py
      ```

      会在ASR目录下生成compiled_encoder.ts compiled_decoder.ts compiled_joiner.ts 三个文件。

## 精度验证

encoder 模型精度验证，屏幕显示Precision test passed 为精度正常。

```shell
python precision_test_onnx.py encoder compiled_encoder.ts
```
decoder 模型精度验证，屏幕显示Precision test passed 为精度正常。
```shell
python precision_test_onnx.py decoder compiled_decoder.ts
```

joiner模型精度验证，屏幕显示Precision test passed 为精度正常。
```shell
python precision_test_onnx.py joiner compiled_joiner.ts
```

## 性能验证
```shell
python perf_test_aie.py \
--encoder_aie_path ./compiled_encoder.ts \
--decoder_aie_path ./compiled_decoder.ts \
--joiner_aie_path ./compiled_joiner.ts \
--device_id 0
```
屏幕上会打印性能数据，以FPS记


## 性能数据 (时延/吞吐率)
|Model| MindIE Torch  | T4| A10|
|------| ----------------- |------| --------|
|encoder| 40.88ms/24.46FPS | 20.53ms/48.70FPS   | 16.4ms/60.9FPS|
|decoder| 0.14ms/6981.55FPS | 0.13ms/7443FPS | 0.12ms/8333FPS |
|joiner | 0.16ms/6186.84FPS | 0.13ms/7612FPS | 0.11ms/9212FPS |