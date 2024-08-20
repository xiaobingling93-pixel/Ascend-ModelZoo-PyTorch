# Zipformer非流式模型-PT推理指导


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

| 配套        | 版本        |
| ----------- | ----------- |
| CANN        | 8.0.T5      |
| Python      | 3.10.13     |
| torch       | 2.1.0       |
| NPU芯片类型 | Ascend310P3 |
| 处理器架构  | arm64       |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 环境安装

1. 安装k2
   
   源码编译。
   
    ```shell
    git clone https://github.com/k2-fsa/k2.git
    cd k2
    export K2_MAKE_ARGS="-j6"
    python3 setup.py install
    ```
    * **若编译失败，尝试再次编译前，需要先删除build文件夹。**
    * 若执行以上命令遇到错误，请参考[此链接](https://k2-fsa.github.io/k2/installation/from_source.html)。    
   
   验证k2是否安装成功  
   
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

## 模型下载
从[此链接](https://huggingface.co/pkufool/icefall-asr-zipformer-wenetspeech-20230615)下载模型相关文件。
模型转换和推理时只需要用到以下文件：  
    - data/lang_char/tokens.txt  
    - exp/epoch-12.pt

下载完后，整理成如下目录结构：
```shell
icefall-asr-zipformer-wenetspeech-20230615
├── data
│   └── lang_char
│       └── tokens.txt
└── exp
    └── epoch-12.pt
```
## 模型推理
1. 打代码补丁(首先将export_onnx.patch与zipformer.patch拷贝到icefall/egs/librispeech/ASR/zipformer目录下)
    ```shell
    cd icefall/egs/librispeech/ASR/zipformer/
    
    patch < export-onnx.patch
    patch < zipformer.patch
    ```
2. 将本代码仓的icefall_pt目录拷贝到icefall工程根目录下。

3. 导出onnx模型与torchscript模型。
    ```shell
   cd icefall/
   repo=/path/to/icefall/egs/librispeech/ASR/icefall-asr-zipformer-wenetspeech-20230615
    # 注意将"icefall-asr-zipformer-wenetspeech-20230615"修改为实际路径
   python ./egs/librispeech/ASR/zipformer/export-onnx.py \
     --tokens $repo/data/lang_char/tokens.txt \
     --use-averaged-model 0 \
     --epoch 12 \
     --avg 1 \
     --exp-dir $repo/exp \
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
     --causal False \
     --chunk-size "16, 32, 64, -1" \
     --left-context-frames "64, 128, 256, -1"
   ```
   执行结束后，会在“icefall-asr-zipformer-wenetspeech-20230615/exp”目录下生成6个模型文件：
    - encoder-epoch-12-avg-1.onnx
    - decoder-epoch-12-avg-1.onnx
    - joiner-epoch-12-avg-1.onnx
    - encoder-epoch-12-avg-1.pt
    - decoder-epoch-12-avg-1.pt
    - joiner-epoch-12-avg-1.pt
4. 对torchscript模型使用PT插件进行编译（分别指定encoder、decoder与joiner的三套参数）。
   ```shell
   cd icefall/icefall_pt
   python export_torch_aie_model.py
   ```
    参数说明：
   ```shell
   --torch_script_path：torhscript模型路径
   --export_part：选择编译部分（encoder，decoder或joiner）
   --soc_version：硬件版本
   --batch_size
   --save_path：编译后的模型保存路径
   ```
    执行结束后，会在save_path对应目录下生成三个编译好的torchscript文件：
    - encoder-epoch-12-avg-1_mindietorch_bs1.pt
    - decoder-epoch-12-avg-1_mindietorch_bs1.pt
    - joiner-epoch-12-avg-1_mindietorch_bs1.pt  

5. 运行推理样例
   ```shell
   cd icefall/icefall_pt
   python pt_val_enc.py
   python pt_val_dec.py
   python pt_val_join.py
   ```
    参数说明：
   ```shell
   --model：PT模型路径
   --need_compile：是否需要编译。若model指向torchscript模型，则需要进行编译后运行
   --soc_version：硬件版本
   --batch_size
   --result_path：模型运行结果保存路径
   --device_id：硬件编号
   --multi：数据加倍的倍数（注意，若multi与batch_size不同时为1，则不生成result文件；测性能部分，脚本默认的warm_up为20loop，若multi不超过20，则不会输出性能）
   ```
6. 精度测试（将onnx_test下的四个测试脚本拷贝到icefall/egs/librispeech/ASR/zipformer）
   ```shell
   cd icefall/egs/librispeech/ASR
   python ./zipformer/onnx_test_enc.py --encoder-model-filename $repo/exp/encoder-epoch-12-avg-1.onnx --tokens $repo/data/lang_char/tokens.txt
   python ./zipformer/onnx_test_dec.py --decoder-model-filename $repo/exp/decoder-epoch-12-avg-1.onnx --tokens $repo/data/lang_char/tokens.txt
   python ./zipformer/onnx_test_join.py --joiner-model-filename $repo/exp/joiner-epoch-12-avg-1.onnx --tokens $repo/data/lang_char/tokens.txt
   ```
    执行结束后，将得到的结果路径与步骤5中得到的结果路径分别配置到脚本cosine_similarity_test.py中，并运行查看相似度：
    ```shell
   python cosine_similarity_test.py
    ```
   可以得出与T4及A10对比的精度结果：
    ```shell
   Cosine Similarity PT_T4_ENC :0.9999+
   Cosine Similarity PT_T4_DEC :1
   Cosine Similarity PT_T4_JOIN :1
   Cosine Similarity PT_A10_ENC :0.9999+
   Cosine Similarity PT_A10_DEC :1
   Cosine Similarity PT_A10_JOIN :1
    ```
7. 性能测试
   1. pt模型性能测试
      第5步运行pt_val脚本时会打印PT模型性能(注意设置multi>20，超过warm_up需要的loop。建议设置多一点)

   2. onnx模型性能测试。  
      第6步运行onnx_test脚本时会打印onnx模型性能

# 模型推理性能精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

Zipformer流式模型由三个子模型组成，分别是encoder、decoder和joiner，其性能如下表所示：

| 模型      | pt插件 - 310P性能（时延/吞吐率）     | T4 onnx性能（时延/吞吐率）         | A10 onnx性能（时延/吞吐率）        |
|---------|---------------------------|---------------------------|---------------------------|
| encoder | 59.5610 ms / 16.7895 fps  | 25.6406 ms / 39.0005 fps  | 16.2751 ms / 61.4434 fps  |
| decoder | 0.4851 ms / 2061.4306 fps | 0.5691 ms / 1757.0740 fps | 0.1219 ms / 8200.5706 fps |
| joiner  | 0.4510 ms / 2217.2949 fps | 0.1526 ms / 6551.7825 fps | 0.1107 ms / 9026.4239 fps |
| 端到端     | 60.4971 ms / 16.5297 fps  | 26.3623 ms /  37.9329 fps | 16.5077 ms / 60.5777 fps  |

