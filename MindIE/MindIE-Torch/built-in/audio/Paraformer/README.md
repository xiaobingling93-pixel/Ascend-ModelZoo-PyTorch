# Paraformer模型-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)

# 概述

该工程使用mindietorch部署Paraformer语音识别模型，同时该工程还适配了VAD音频切分模型以及PUNC标点符号模型，三个模型可组成VAD+Paraformer+PUNC的pipeline，实现对于长音频的识别

- 模型路径:
    ```bash
    https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
    ```

- 参考实现：
    ```bash
    https://github.com/modelscope/FunASR
    ```

# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套   | 版本    | 环境准备指导 |
  | ------ | ------- | ------------ |
  | Python | 3.10.13 | -            |
  | torch  | 2.1.0+cpu   | -            |
  | torch_audio  | 2.1.0+cpu   | -            |
  | CANN   | 8.0.RC3 | -            |
  | MindIE | 1.0.RC3 | -       |

# 快速上手
## 获取源码

1. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```

2. 获取Funasr源码

    ```
    git clone https://github.com/modelscope/FunASR.git
    cd ./FunASR
    git reset fdac68e1d09645c48adf540d6091b194bac71075 --hard
    cd ..
    ```

3. 修改Funasr的源码，将patch应用到代码中（若patch应用失败，则需要手动进行修改）
    ```
    cd ./FunASR
    git apply ../mindie.patch --ignore-whitespace
    cd ..
    ```

    （可选）若为Atlas 800I A2服务器，可以使用如下命令将Attention替换为Flash Attention，可以提升Paraformer模型的性能
    ```
    cd ./FunASR
    git apply ../mindie_fa.patch --ignore-whitespace
    cd ..
    ```

4. 获取模型文件

    将[Paraformer](https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)的模型文件下载到本地，并保存在./model文件夹下

    将[vad](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/files)的模型文件下载到本地，并保存在./model_vad文件夹下

    将[punc](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/files)的模型文件下载到本地，并保存在./model_punc文件夹下

    目录结构如下所示

    ```
    Paraformer
    ├── FunASR
    └── model
        └── model.pt
        └── config.yaml
        └── ...
    └── model_vad
        └── model.pt
        └── config.yaml
        └── ...
    └── model_punc
        └── model.pt
        └── config.yaml
        └── ...
    ```

5. 安装Funasr的依赖
    ```
    apt install ffmpeg
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    pip install jieba omegaconf kaldiio librosa tqdm hydra-core six attrs psutil tornado
    ```

6. 安装配套版本的torch_npu，同时参考[昇腾文档](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindietorch/Torchdev/mindie_torch0018.html)兼容mindie和torch_npu

7. (可选) 若要进行精度或性能测试，可下载数据集[AISHELL-1](https://www.aishelltech.com/kysjcp)并保存于任意路径

# 模型编译
1. （可选）模型序列化
    若CPU为aarch64的架构，则在编译encoder和decoder时会出现RuntimeError: could not create a primitive descriptor for a matmul primitive，此时需要新创建一个Python环境（推荐使用conda创建），使用如下命令安装torch 2.2.1及相关依赖
    ```
    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
    pip install omegaconf kaldiio librosa tqdm hydra-core six
    ```

    而后，执行如下脚本将encoder和decoder序列化
    ```bash
    python trace_encoder_decoder.py \
    --model ./model \
    --traced_encoder ./compiled_model/traced_encoder.pt \
    --traced_decoder ./compiled_model/traced_decoder.pt
    ```
    参数说明：
      - --model：预训练模型路径
      - --traced_encoder: 序列化后的encoder保存路径
      - --traced_decoder: 序列化后的decoder保存路径
    
    该步骤获得序列化后的encoder和decoder模型，后续模型编译仍需回到原始环境


2. 模型编译
   执行下述命令进行模型编译（如编译后的模型保存于compiled_model目录下，需要首先mkdir compiled_model）
    ```bash
    python compile.py \
    --model ./model \
    --model_vad ./model_vad \
    --model_punc ./model_punc \
    --compiled_encoder ./compiled_model/compiled_encoder.pt \
    --compiled_decoder ./compiled_model/compiled_decoder.pt \
    --compiled_cif ./compiled_model/compiled_cif.pt \
    --compiled_cif_timestamp ./compiled_model/compiled_cif_timestamp.pt \
    --compiled_vad ./compiled_model/compiled_vad.pt \
    --compiled_punc ./compiled_model/compiled_punc.pt \
    --traced_encoder ./compiled_model/traced_encoder.pt \
    --traced_decoder ./compiled_model/traced_decoder.pt \
    --soc_version Ascendxxx
    ```

    参数说明：
      - --model：预训练模型路径
      - --model_vad：VAD预训练模型路径，若不使用VAD模型则设置为None
      - --model_punc：PUNC预训练模型路径，若不使用PUNC模型则设置为None
      - --compiled_encoder：编译后的encoder模型的保存路径
      - --compiled_decoder：编译后的decoder模型的保存路径
      - --compiled_cif：编译后的cif函数的保存路径
      - --compiled_cif_timestamp：编译后的cif_timestamp函数的保存路径
      - --compiled_vad：编译后的vad的保存路径
      - --compiled_punc：编译后的punc的保存路径
      - --traced_encoder：预先序列化的encoder模型的路径，若并未执行第2步提前编译模型，则无需指定该参数
      - --traced_decoder：预先序列化的decoder模型的路径，若并未执行第2步提前编译模型，则无需指定该参数
      - --soc_version：昇腾芯片的型号，需要执行npu-smi info命令进行查询，并在查询到的“Name”前增加Ascend字段，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。


## 模型推理
1. 设置mindie内存池上限为12G，执行如下命令设置环境变量
    ```
    export INF_NAN_MODE_ENABLE=0
    export TORCH_AIE_NPU_CACHE_MAX_SIZE=12
    ```

2. 样本测试
   执行下述命令进行音频样本测试，该脚本将会测试VAD+Paraformer+PUNC整个Pipeline，脚本单次只会读取一个音频文件，音频文件可以为任意长度
    ```bash
    python test.py \
    --model ./model \
    --model_vad ./model_vad \
    --model_punc ./model_punc \
    --compiled_encoder ./compiled_model/compiled_encoder.pt \
    --compiled_decoder ./compiled_model/compiled_decoder.pt \
    --compiled_cif ./compiled_model/compiled_cif.pt \
    --compiled_cif_timestamp ./compiled_model/compiled_cif_timestamp.pt \
    --compiled_punc ./compiled_model/compiled_punc.pt \
    --compiled_vad ./compiled_model/compiled_vad.pt \
    --paraformer_batch_size 16 \
    --sample_path ./model/example \
    --soc_version Ascendxxx
    ```

    参数说明：
      - --model：预训练模型路径
      - --model_vad：VAD预训练模型路径
      - --model_punc：PUNC预训练模型路径
      - --compiled_encoder：编译后的encoder模型的路径
      - --compiled_decoder：编译后的decoder模型的路径
      - --compiled_cif：编译后的cif函数的路径
      - --compiled_cif_timestamp：编译后的cif_timestamp函数的路径
      - --compiled_punc：编译后的punc模型的路径
      - --compiled_vad：编译后的vad模型的路径
      - --paraformer_batch_size：Paraformer模型所使用的batch_size
      - --sample_path：测试音频的路径或所在的文件夹路径，若为文件夹路径则会遍历文件夹下的所有音频文件
      - --soc_version：昇腾芯片的型号，需要执行npu-smi info命令进行查询，并在查询到的“Name”前增加Ascend字段，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。

3. 性能测试
    执行下述命令对于Paraformer进行性能测试，该脚本仅针对Paraformer模型进行测试，batch_size参数用于控制同时处理的最大音频数量（例如设置为64，则会在sample_path下同时读取64个音频，并组合成一个输入进行处理），但需要注意音频的长度不能过长，否则可能超出NPU的显存
    ```
    python test_performance.py \
    --model ./model \
    --compiled_encoder ./compiled_model/compiled_encoder.pt \
    --compiled_decoder ./compiled_model/compiled_decoder.pt \
    --compiled_cif ./compiled_model/compiled_cif.pt \
    --compiled_cif_timestamp ./compiled_model/compiled_cif_timestamp.pt \
    --batch_size 64 \
    --result_path ./aishell_test_result.txt \
    --sample_path /path/to/AISHELL-1/wav/test \
    --soc_version Ascendxxx
    ```

    参数说明：
      - --model：预训练模型路径
      - --compiled_encoder：编译后的encoder模型的路径
      - --compiled_decoder：编译后的decoder模型的路径
      - --compiled_cif：编译后的cif函数的路径
      - --compiled_cif_timestamp：编译后的cif_timestamp函数的路径
      - --batch_size：Paraformer模型所使用的batch_size
      - --sample_path：AISHELL-1测试集音频所在路径，模型会递归查找该路径下的所有音频文件
      - --result_path：测试音频的推理结果的保存路径
      - --soc_version：昇腾芯片的型号，需要执行npu-smi info命令进行查询，并在查询到的“Name”前增加Ascend字段，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。
      

4. 精度测试

    利用如下命令安装中文文本精度对比库nltk
    ```
    pip install nltk
    ```

    需要首先执行第4步完成性能测试，而后利用性能测试保存到result_path的结果进行精度验证，执行如下命令
    ```
    python test_accuracy.py \
    --result_path ./aishell_test_result.txt \
    --ref_path /path/to/AISHELL-1/transcript/aishell_transcript_v0.8.txt
    ```

    参数说明：
      - --result_path：测试音频的推理结果的保存路径
      - --ref_path：AISHELL-1测试音频的Ground Truth所在路径


## 模型精度及性能

该模型在Atlas 300I pro和Atlas 800I A2上的参考性能及精度如下所示（其中性能数据为Paraformer模型纯推理性能，并非端到端推理性能）

| NPU            | batch_size | rtx_avg | cer     |
|----------------|------------|---------|---------|
| Atlas 300I pro | 16         | 217.175 | 0.0198  |
| Atlas 800I A2  | 64         | 461.775 | 0.0198  |