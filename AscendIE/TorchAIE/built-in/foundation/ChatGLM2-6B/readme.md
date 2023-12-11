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



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.0|
  | CANN                                                         | 7.0.0 B050 | -                                                            |
  | Python                                                       | 3.9.0   | -                                                            |
  | PyTorch                                                      | 2.1.0   | -                                                            |
  | Torch_AIE                                                    | 6.3.rc2   | 


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取代码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/TorchAIE/built-in/foundation/ChatGLM2-6B       
   ```

2. 安装依赖。

   ```
   pip3 install -r requirement.txt
   需要安装pt插件的python wheel(可根据代码仓中的readme.md操作) 和统一接口的run包。
   参考：
   https://gitee.com/ascend/ascend-inference-ptplugin.git
   ```

   #### 安装推理引擎

   ```
   chmod +x Ascend-cann-aie_7.0.T51.B010_linux-aarch64.run
   Ascend-cann-aie_7.0.T51.B010_linux-aarch64.run --install
   cd Ascend-cann-aie
   source set_env.sh
   ```

   #### 安装torch_aie

   ```
   tar -zxvf Ascend-cann-torch-aie-6.3.RC2-linux_aarch64.tar.gz
   pip3 install torch-aie-6.3.RC2-linux_aarch64.whl
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
   python3.9 compile_model.py
   ```  

2. 开始对话验证。
   ```
   python3 example.py
   ```
   