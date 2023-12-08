# ChatGLM2-6B 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型编译与推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ChatGLM2-6B 是开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 拥有更强大的性能和更长的上下文以及更高效的推理。
- 参考实现 ：
  ```
  url=https://github.com/THUDM/ChatGLM2-6B
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_ids    | INT64 | batchsize x 512 | ND         |
  | position_ids    | INT64 | batchsize x 512 | ND         |
  | attention_mask    | INT64 | batchsize x 512 | ND         |
  | past_key_values    | FLOAT | batchsize x 512 | ND         |
  


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | lm_logits  | FLOAT  | batchsize x 512 x 21128 | ND           |
  | past_key_values  | FLOAT  | batchsize x 512 x 21128 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.0|
  | CANN                                                         | 7.0.0 B050 | -                                                            |
  | Python                                                       | 3.9.0   | -                                                            |
  | PyTorch                                                      | 2.1.0   | -                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/TorchAIE/built-in/foundation/ChatGLM2-6B       
   ```

2. 安装依赖。

   ```
   pip3 install -r requirement.txt
   ```

3. 获取权重和配置文件
   
   用户可以从[这里](https://huggingface.co/THUDM/chatglm2-6b/tree/dev)下载预训练权重和配置文件，然后将这些文件放在 "model"文件夹中，**不要覆盖 `modeling_chatglm.py`文件**。
   `model`文件夹内容如下：
   ```shell
   ├── model
         ├──config.json
         ├──configuration_chatglm.py
         ├──pytorch_model-00001-of-00007.bin
         ├──pytorch_model-00002-of-00007.bin
         ├──pytorch_model-00003-of-00007.bin
         ├──pytorch_model-00004-of-00007.bin
         ├──pytorch_model-00005-of-00007.bin
         ├──pytorch_model-00006-of-00007.bin
         ├──pytorch_model-00007-of-00007.bin
         ├──pytorch_model.bin.index.json
         ├──quantization.py
         ├──tokenization_chatglm.py
         ├──tokenizer_config.json
         ├──tokenizer.model
         ├──modeling_chatglm.py
   ```

## 模型编译与推理<a name="section741711594517"></a>

1. 模型编译。

   使用torch aie将模型权重文件pytorch_model.bin，trace为pt文件，然后保存为.ts文件。
   ```
   python3.9 compile_model.py --batch_size=1
   ```  

2. 开始对话验证。
   ```
   python3 example.py
   ```
   
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度指标（Loss）| 性能 |
| :------: | :--------: | :----: | :--: | :--: |


> 注：衡量精度的指标为验证集平均交叉熵损失（Cross-Entropy Loss），数值越低越好。

##  准备评估精度数据集<a name="section741711594517"></a>

用户可以从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0) 下载处理好的 C-Eval 数据集，解压到 `evaluation` 目录下。

## 运行评估任务<a name="section741711594517"></a>
1）首先修改评估脚本`evaluation/evaluate_ceval.py`。