# Zipformer非流式模型-OM推理指导


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
| CANN                        | 8.0.T5      | -                                                       |
| Python                      | 3.10.13     |                                                           
| torch                       | 2.1.0       |
| 芯片类型                        | 300I PRO | -    

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
    cd icefall
    git reset --hard e2fcb42f5f176d9e39eb38506ab99d0a3adaf202

    pip install -r requirements.txt
    ```
4. 安装onnx改图工具
    ```shell
    git clone https://gitee.com/ascend/msadvisor.git
    cd msadvisor/auto-optimizer
    python3 -m pip install .
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
   git clone https://huggingface.co/pkufool/icefall-asr-zipformer-wenetspeech-20230615
    ```
   若下载失败，请尝试从以上链接手动下载文件。模型转换和推理时只需要用到以下文件：
    - data/lang_char/tokens.txt
    - exp/epoch-12.pt

## 模型转换
1. 打代码补丁(首先将export_onnx.patch与zipformer.patch拷贝到icefall/egs/librispeech/ASR/zipformer目录下)
    ```shell
    cd icefall/egs/librispeech/ASR/zipformer/
    
    patch < export-onnx.patch
    patch < zipformer.patch
    ```

2. 导出onnx模型与torchscript模型
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
   执行结束后，会在“icefall-asr-zipformer-wenetspeech-20230615/exp”目录下生成6个文件：
    - encoder-epoch-12-avg-1.onnx
    - decoder-epoch-12-avg-1.onnx
    - joiner-epoch-12-avg-1.onnx
    - encoder-epoch-12-avg-1.pt
    - decoder-epoch-12-avg-1.pt
    - joiner-epoch-12-avg-1.pt

3. 使用onnxsim进行常量折叠
   ```shell
   cd icefall-asr-zipformer-wenetspeech-20230615/exp
   
   onnxsim encoder-epoch-12-avg-1.onnx encoder-epoch-12-avg-1_sim_py310.onnx
   onnxsim decoder-epoch-12-avg-1.onnx decoder-epoch-12-avg-1_sim_py310.onnx
   ```
   
4. 使用本代码仓的 `modify_encoder.py`与`modify_decoder.py`进行改图，以便正确导出om
    ```shell
   python modify_encoder.py --onnx ./encoder-epoch-12-avg-1_sim_py310.onnx 
   python modify_decoder.py --onnx ./decoder-epoch-12-avg-1_sim_py310.onnx
    ```
    执行结束后，会在`icefall-asr-zipformer-wenetspeech-20230615/exp`目录下生成`encoder-epoch-12-avg-1_sim_py310_modified.onnx`
与`decoder-epoch-12-avg-1_sim_py310_modified.onnx`文件。

5. 执行以下命令导出om  
    ```shell
    atc --input_shape="x:1,100,80;x_lens:1" --precision_mode=force_fp32 --soc_version=Ascend310P3 --framework=5 --output=encoder_py310 --model=./encoder-epoch-12-avg-1_sim_py310_modified.onnx
    atc --input_shape="y:1,2" --precision_mode=force_fp32 --soc_version=Ascend310P3 --framework=5 --output=decoder_py310 --model=./decoder-epoch-12-avg-1_sim_py310_modified.onnx
    atc --input_shape="encoder_out:1,512;decoder_out:1,512" --soc_version=Ascend310P3 --framework=5 --output=joiner_py310 --model=./joiner-epoch-12-avg-1.onnx
   ```
    执行结束后，会在当前目录下生成三个om文件：
    - encoder_py310_linux_aarch64.om
    - decoder_py310.om
    - joiner_py310.om

## 模型推理
1. 运行推理样例（首先将代码create_data.py拷贝到icefall/egs/librispeech/ASR/icefall-asr-zipformer-wenetspeech-20230615/exp下）
   ```shell
   cd icefall/egs/librispeech/ASR/icefall-asr-zipformer-wenetspeech-20230615/exp
   python create_data.py
   python -m ais_bench --model ./encoder_py310_linux_aarch64.om --input "./x.npy,./x_lens.npy" --dymShape "x:1,100,80;x_lens:1" --outputSize 1000000,1000000 --output /path/to/save/result --output_dirname ./encoder --outfmt TXT --warmup_count 20 --loop 2000
   python -m ais_bench --model ./decoder_py310.om --input "./y.npy" --output /path/to/save/result --output_dirname ./decoder --outfmt TXT --warmup_count 20 --loop 2000
   python -m ais_bench --model ./joiner_py310.om --input "./encoder_out.npy,./decoder_out.npy" --output /path/to/save/result --output_dirname ./joiner --outfmt TXT --warmup_count 20 --loop 2000
   ```
2. 精度测试（将onnx_test下的四个测试脚本拷贝到icefall/egs/librispeech/ASR/zipformer）
   ```shell
   repo=/path/to/icefall/egs/librispeech/ASR/icefall-asr-zipformer-wenetspeech-20230615
   cd icefall/egs/librispeech/ASR
   python ./zipformer/onnx_test_enc.py --encoder-model-filename $repo/exp/encoder-epoch-12-avg-1.onnx --tokens $repo/data/lang_char/tokens.txt
   python ./zipformer/onnx_test_dec.py --decoder-model-filename $repo/exp/decoder-epoch-12-avg-1.onnx --tokens $repo/data/lang_char/tokens.txt
   python ./zipformer/onnx_test_join.py --joiner-model-filename $repo/exp/joiner-epoch-12-avg-1.onnx --tokens $repo/data/lang_char/tokens.txt
   ```
    执行结束后，将得到的结果路径与步骤1中得到的结果路径分别配置到脚本cosine_similarity_test.py中，并运行查看相似度：
    ```shell
   python cosine_similarity_test.py
    ```
   可以得出与T4及A10对比的精度结果：
    ```shell
   Cosine Similarity OM_T4_ENC :0.9999+
   Cosine Similarity OM_T4_DEC :0.9999+
   Cosine Similarity OM_T4_JOIN :0.9999+
   Cosine Similarity OM_A10_ENC :0.9999+
   Cosine Similarity OM_A10_DEC :0.9999+
   Cosine Similarity OM_A10_JOIN :0.9999+
    ```
3. 性能测试
   1. om模型性能测试
      
      第1步运行om时会打印模型性能
   2. onnx模型性能测试。  
      第2步运行onnx_test脚本时会打印onnx模型性能
# 模型推理性能精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

Zipformer流式模型由三个子模型组成，分别是encoder、decoder和joiner，其性能如下表所示：

| 模型      | OM - 300I PRO性能（时延/吞吐率）        | T4 onnx性能（时延/吞吐率）         | A10 onnx性能（时延/吞吐率）        |
|---------|----------------------------|---------------------------|---------------------------|
| encoder | 44.2362 ms / 22.6059 fps   | 25.6406 ms / 39.0005 fps  | 16.2751 ms / 61.4434 fps  |
| decoder | 0.0613 ms / 16300.6119 fps | 0.5691 ms / 1757.0740 fps | 0.1219 ms / 8200.5706 fps |
| joiner  | 0.0669 ms / 14930.0412 fps | 0.1526 ms / 6551.7825 fps | 0.1107 ms / 9026.4239 fps |
| 端到端     | 44.3644 ms / 22.5405 fps   | 26.3623 ms /  37.9329 fps | 16.5077 ms / 60.5777 fps  |
