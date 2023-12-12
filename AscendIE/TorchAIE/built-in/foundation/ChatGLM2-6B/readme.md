# ChatGLM2-6B 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型编译与推理](#section741711594517)


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
   我们依赖transformers, pytorch， TorchAIE和统一接口，以及一些原本chatGLM2-6B依赖的一些包，具体如下：

   #### 安装非推理引擎的依赖
   ```
   pip3 install -r requirement.txt
   ```
   需要安装pt插件的python wheel(可根据代码仓中的readme.md操作) 和统一接口的run包。
   参考：
   https://gitee.com/ascend/ascend-inference-ptplugin.git

   这个时候我们可以通过命令`pip show torch`找到torch的目录， 比如'/usr/local/python3/lib/python3.9/site-packages/torch', 这个路径我们定义为${TORCH_ROOT_PATH}, 后续C++编译中需要用到。

   #### 安装推理引擎统一接口

   ```
   chmod +x Ascend-cann-aie_7.0.T51.B010_linux-aarch64.run
   Ascend-cann-aie_7.0.T51.B010_linux-aarch64.run --install
   cd /usr/local/Ascend/aie/
   source set_env.sh
   ```
   

   #### 安装torch_aie

   ```
   tar -zxvf Ascend-cann-torch-aie-${version}-linux_aarch64.tar.gz
   pip3 install torch-aie-${version}-linux_aarch64.whl
   ```
   这个时候我们可以通过`pip show torch_aie`找到torch_aie的目录， 比如'/usr/local/python3/lib/python3.9/site-packages/torch_aie', 这个路径我们定义为${TORCH_AIE_PATH}, 后续C++编译中需要用到。

   

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
注意：如果设置的device_id不为0， 那么需要做一下操作。
```
1 `sed -i 's/npu:0/npu:${device_id}' model/modeling_chatglm.py ` 将device_id换成自己定义的id.
2  将 run.sh里头的 `./sample 0` 替换为 `./sample ${device_id}`
```
1. trace模型与模型编译。

   使用torch aie将模型源码，trace为pt文件。
   ```
   python3.9 compile_model.py --device 0
   ```  
   compile_model的参数和默认值如下
   ```
   --device 0 \  # 环境使用的device_id
   --pretrained_model ./model/ # 源码和权重文件落盘位置
   --need_trace true # 是否需要trace 
   ```
   ```
   模型编译`compile`文件夹内容如下：
   ```shell
   ├── compile
         ├──build.sh
         ├──chatglm2_test.cpp
         ├──CMakeLists
         ├──run.sh
   ```

   进行 C++编译，进入compile目录。 修改CMakeLists中17行的`${TORCH_AIE_PATH}`修改为 快速上手/安装torch_aie里提到的
   `${TORCH_AIE_PATH}`, 将20行`${TORCH_ROOT_PATH}`替换为 快速上手/安装依赖中的`${TORCH_ROOT_PATH}`。
   将模型编译为ts文件。
   ```
   bash build.sh
   bash run.sh
   ```



2. 开始对话验证。
   ```
   python3 example.py --device 0
   ```

3. 最后对话的效果如下
```
欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序

用户：晚上睡不着应该怎么办

ChatGLM：以下时是一些有助于放松和入睡的技巧:

1. 创建一个放松的睡眠环境。确保房间安静、黑暗、凉爽和舒适。

2. 创造一个睡前惯例。例如,洗澡,做一些伸展运动,或者喝一杯温热的饮料。

3. 避免在睡前吃大量的食物或饮料。特别是咖啡因和酒精。

4. 避免在睡前使用电子设备。手机,电脑和电视等蓝光屏幕会刺激大脑,使你难以入睡。

5. 适当的运动。但是请避免在睡前激烈运动。

6. 放松技巧。例如冥想,深呼吸或渐进性肌肉放松。

如果这些技巧无法解决你的问题,你可以考虑咨询医生或睡眠专家。
```
   
