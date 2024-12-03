# Conformer模型- OM推理指导


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
  | CANN                        | 8.0.RC1     |
  | Python                      | 3.10.13     |
  | torch                       | 2.1.0       |
  | torchaudio                    | 2.1.0         |
  | onnxsim                        | 0.4.36|
  | 支持产品                     | Atlas 300I Pro推理卡 |
  | 处理器架构 | arm64 |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 环境安装

1. 安装k2
   
   源码编译。
   
    ```shell
    git clone https://github.com/k2-fsa/k2.git
    cd k2
    export K2_MAKE_ARGS="-j"
    python3 setup.py install
    ```
   若执行以上命令遇到错误，请参考[此链接](https://k2-fsa.github.io/k2/installation/from_source.html)。
   
   验证k2是否安装成功  
   
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

3. 安装auto_optimizer

    参考[链接](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)中的说明进行安装

4. 安装icefall

    ```shell
    git clone https://github.com/k2-fsa/icefall.git
    git reset --hard e2fcb42f5f176d9e39eb38506ab99d0a3adaf202
       
    cd icefall
    pip install -r requirements.txt
    ```

5. 将icefall加入环境变量, "/path/to/icefall"替换为icefall文件夹所在的路径。
   **这一步很重要，否则会报icefall找不到的错误。**

    ```shell
    export PYTHONPATH=/path/to/icefall:$PYTHONPATH
    ```

6. 安装AIS_BENCH工具

    参考[链接](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)中的说明进行安装

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
   precision_test_om.py
   modify_decoder_onnx.py
   modify_encoder_onnx.py
   data_gen.py
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
├── precision_test_om.py
├── modify_decoder_onnx.py
├── modify_encoder_onnx.py
└── data_gen.py
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

   导出后，Exp目录下会有encoder-epoch-99-avg-1.onnx
   decoder-epoch-99-avg-1.onnx, joiner-epoch-99-avg-1.onnx三个文件。

4. ONNX模型简化

   对于encoder和decoder，还需要执行ONNXSIM进行模型简化

   ```sh
   onnxsim ./exp/encoder-epoch-99-avg-1.onnx ./simed_encoder.onnx
   ```

   ```sh
   onnxsim ./exp/decoder-epoch-99-avg-1.onnx ./simed_decoder.onnx
   ```

5. ONNX转换om模型

   1. 对于encoder，需要先进行改图

      ```sh
      python modify_encoder_onnx.py ./simed_encoder.onnx
      ```

      再进行om转换，其中参数soc_version用于指定模型转换时昇腾AI处理器的版本（需要执行npu-smi info命令进行查询，并在查询到的“Name”前增加Ascend字段，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy）

      ```sh
      atc --input_shape="x:1,100,80;x_lens:1" --precision_mode=force_fp32 --soc_version=Ascendxxxyy --framework=5 --output=encoder_py310 --model ./simed_encoder_modified.onnx
      ```

   2. 对于decoder，需要先进行改图

      ```sh
      python modify_decoder_onnx.py ./simed_decoder.onnx
      ```

      再进行om转换，其中参数soc_version用于指定模型转换时昇腾AI处理器的版本（需要执行npu-smi info命令进行查询，并在查询到的“Name”前增加Ascend字段，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy）

      ```sh
      atc --input_shape="y:1,2" --precision_mode=force_fp32 --soc_version=Ascendxxxyy --framework=5 --output=decoder_py310 --model=simed_decoder_modified.onnx
      ```

   3. Joiner模型可以直接转换om，其中参数soc_version用于指定模型转换时昇腾AI处理器的版本（需要执行npu-smi info命令进行查询，并在查询到的“Name”前增加Ascend字段，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy）

      ```sh
      atc --input_shape="encoder_out:1,512;decoder_out:1,512" --soc_version=Ascendxxxyy --framework=5 --output=joiner_py310 --model=./exp/joiner-epoch-99-avg-1.onnx
      ```

6. 导出ts模型

   请查看代码仓[Conformer模型-推理指导](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/audio/Conformer)的README.md将ONNX模型转换为ts模型。

## 性能测试

encoder模型性能测试：

```sh
python data_gen.py
```

```shell
python -m ais_bench --model ./encoder_py310_linux_aarch64.om --input "x.npy,x_lens.npy" --dymShape "x:1,100,80;x_lens:1" --outputSize 1000000,1000000  --loop 100
```

decoder模型性能测试
```shell
python -m ais_bench --model ./decoder_py310.om --loop 1000
```

joiner模型性能测试
```shell
python -m ais_bench --model ./joiner_py310.om --loop 2000
```
## 精度测试
encoder精度测试，脚本执行后显示Precision test passed为通过
```shell
python precision_test_om.py encoder ./encoder_py310_linux_aarch64.om ./compiled_encoder.ts
```

decoder精度测试，脚本执行后显示Precision test passed为通过
```shell
python precision_test_om.py decoder ./decoder_py310.om ./compiled_decoder.ts
```

joiner精度测试，脚本执行后显示Precision test passed为通过
```shell
python precision_test_om.py joiner ./joiner_py310.om ./compiled_joiner.ts
```

### 性能数据
|Model| OM                  | T4| A10|
|------|---------------------|------| --------|
|encoder| 37.65ms/26.56FPS    | 20.53ms/48.70FPS   | 16.4ms/60.9FPS|
|decoder| 0.048ms/20886.41FPS | 0.13ms/7443FPS | 0.12ms/8333FPS |
|joiner | 0.054ms/18456.47FPS | 0.13ms/7612FPS | 0.11ms/9212FPS |