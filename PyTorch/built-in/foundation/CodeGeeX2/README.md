# codegeex2_NPU

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

CodeGeeX2 是多语言代码生成模型 [CodeGeeX](https://github.com/THUDM/CodeGeeX) ([KDD’23](https://arxiv.org/abs/2303.17568)) 的第二代模型。不同于一代 CodeGeeX（完全在国产华为昇腾芯片平台训练） ，CodeGeeX2 是基于 [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) 架构加入代码预训练实现，得益于 ChatGLM2 的更优性能，CodeGeeX2 在多项指标上取得性能提升（+107% > CodeGeeX；仅60亿参数即超过150亿参数的 StarCoder-15B 近10%），更多特性包括：

- **更强大的代码能力**：基于 ChatGLM2-6B 基座语言模型，CodeGeeX2-6B 进一步经过了 600B 代码数据预训练，相比一代模型，在代码能力上全面提升，[HumanEval-X](https://huggingface.co/datasets/THUDM/humaneval-x) 评测集的六种编程语言均大幅提升 (Python +57%, C++ +71%, Java +54%, JavaScript +83%, Go +56%, Rust +321%)，在Python上达到 35.9% 的 Pass@1 一次通过率，超越规模更大的 StarCoder-15B。
- **更优秀的模型特性**：继承 ChatGLM2-6B 模型特性，CodeGeeX2-6B 更好支持中英文输入，支持最大 8192 序列长度，推理速度较一代 CodeGeeX-13B 大幅提升，量化后仅需6GB显存即可运行，支持轻量级本地化部署。
- **更全面的AI编程助手**：CodeGeeX插件（[VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex), [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex)）后端升级，支持超过100种编程语言，新增上下文补全、跨文件补全等实用功能。结合 Ask CodeGeeX 交互式AI编程助手，支持中英文对话解决各种编程问题，包括且不限于代码解释、代码翻译、代码纠错、文档生成等，帮助程序员更高效开发。
- **更开放的协议**：CodeGeeX2-6B 权重对学术研究完全开放，填写[登记表](https://open.bigmodel.cn/mla/form?mcode=CodeGeeX2-6B)申请商业使用。



- 参考实现：

  ```
  url=https://github.com/THUDM/ChatGLM2-6B
  commit_id=877ef10d85c93ddfcfe945fcdc764393a52541b8
  url=https://huggingface.co/THUDM/chatglm2-6b/tree/v1.0
  commit_id=0ade0d38ac00258ae09450696315c2ff0b1faf12
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```


# 准备训练环境

该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
2. 查看[软件版本配套表](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitee.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**

## 准备环境

默认配置需要每张卡有60G以上空闲内存。

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version | 三方库依赖版本  |
  | :-----------: | :-------------: |
  | PyTorch 1.11  | deepspeed 0.9.2 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  1. 安装基础依赖

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

  ```
  pip install -r requirements.txt  # PyTorch1.11版本
  ```

  2. 安装deepspeed_npu插件

  ```
  # v0.9.2分支
  pip3 install deepspeed==0.9.2
  git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
  cd deepspeed_npu
  git checkout 5c7c89930f0b70ea586d5db63f8e66477d5d9d9f
  pip3 install .
  ```

  3. 替换transformers依赖文件

  ```
   # 自动替换无法替换三方库中的文件。
   pip show transformers
   # 获取transformers的Location路径
   # 使用fix文件夹下的tranining_args.py替换路径下transformers/tranining_args.py
   规避保存权重问题：
   # 使用fix文件夹下的modeling_utils.py替换transformers/modeling_utils.py
  ```


## 准备数据集

1. 获取数据集。

   code_alpaca_20k

   

   ```
   ├── ptuning
         ├──train.json
         ├──dev.json
   ```

2. 预处理数据集。
   为了方便启动训练后，不用再每次重复加载处理数据集，故提前进行处理。也可以下载提前处理好的[数据集](待补充）

```shell
bash preprocess.sh
```

处理好的数据集位于同目录下的train_datasets文件夹下，参考目录如下

```
   ├── train_datasets
         ├──data-00000-of-00003.arrow
         ├──data-00001-of-00003.arrow
         ├──data-00002-of-00003.arrow 
         ├──dataset_info.json
         ├──state.json
```


## 准备模型权重

1. 获取语言识别模型和预训练权重

   1)用户从[链接]([THUDM/codegeex2-6b at main (huggingface.co)](https://huggingface.co/THUDM/codegeex2-6b/tree/main))自行获取模型文件（除了modeling_chatglm.py）和权重文件（pytorch_model-0000*-of-00007.bin），并放于model目录下，微调依赖该模型权重。  

   2)Please Do NOT overwrite modeling_chatglm.py

     The "model" directory is as follows

   ```shell
     ├── model
         ├──config.json
         ├──configuration_chatglm.py
         ├──ice_text.model
         ├──pytorch_model-00001-of-00007.bin
         ├──pytorch_model-00002-of-00007.bin
         ├──pytorch_model-00003-of-00007.bin
         ├──pytorch_model-00004-of-00007.bin
         ├──pytorch_model-00005-of-00007.bin
         ├──pytorch_model-00006-of-00007.bin
         ├──pytorch_model-00007-of-00007.bin
         ├──pytorch_model.bin.index.json
         ├──quantization.py
         ├──test_modeling_chatglm.py
         ├──tokenization_chatglm.py
         ├──tokenizer_config.json
         ├──tokenizer.model
         ├──modeling_chatglm.py
   ```

   

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}/ptuning
   ```

   修改ptuning目录下的env_npu.sh，修改引用的环境变量位置。

2. 运行训练脚本。

   该模型P-Tuning v2支持单机单卡，全参数fintune支持单机8卡。

   - P-Tuning v2

     启动P-Tuning v2。

     ```
     bash train.sh
     ```

   - 全参数finetune

     启动8卡微调。
     可以用deepspeed.json配置deepspeed参数，目前默认使用zero2

     ```
     bash ds_train_fintune.sh
     ```


   模型训练参数说明如下。

   ```
   公共参数：
   --max_source_length                       //处理后句子长度
   --max_target_length                       //目标数据长度
   --per_device_train_batch_size             //每卡训练批次大小
   --gradient_accumulation_steps             //梯度更新步数
   --max_steps                               //最大训练步数
   --logging_steps                           //打印信息步数
   --save_steps                              //保存参数步数
   --learning_rate                           //学习率
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练相关信息。

## 验证模型

1. 全参数finetune验证

   运行以下命令

   ```
   cd /${模型文件夹名称}/scripts
   ./run_humanevalx.sh
   ```

   生成结果在屏幕上显示

# 训练结果展示

**表 1**  训练结果展示表


|     NAME      | SamplesPerSec | Iterations | DataType | Torch_Version | Card |
| :-----------: | :-----------: | :--------: | :------: | :-----------: | :--: |
| Finetune -NPU |   (待补充)    |  (待补充)  |  bf16?   |     1.11      | Atlas 900 A2 PODc |
| Finetune -GPU |   (待补充)    |  (待补充)  |  bf16?   |     1.11      | 竞品A |

说明：P-Tuning 仅打通功能，无性能优化。

**表 2**  评估结果展示表

| 评估项  | NPU  | GPU  |
| :-----: | :--: | :--: |
| human pass@1  |  0.37    |    0.35  |


说明：该结果是step=100的验证结果。

# 版本说明

## 变更

2023.9.1：首次发布。

## FAQ

1. 报错提示deepspeed.py需要版本大于等于0.6.5

   ```
   # 关闭版本检测（如安装0.9.2版本无需此操作）
   # 若遇到该报错
   pip show transformers
   # 复制Location路径
   # 使用fix文件夹下的deepspeed.py替换路径下transformers/deepspeed.py
   ```

2. 加载参数阶段有卡死现象

   ```
   删除root下的cache目录，重新运行
   ```

3. 单卡阶段报embedding_dense_grad算子错误

   ```
   enbedding当前版本，不支持动静合一，静态有部分shape不支持,新版本已修复
   # 若遇到该报错
   修改main.py文件
   torch.npu.set_compile_mode(jit_compile=False)
   ```

4. 提示so文件错误

   ```
   提示so文件找不到
   # 若遇到该报错
   全局搜索so的位置，然后导入环境变量
   export LD_LIBRARY_PATH=/usr/:$LD_LIBRARY_PATH
   ```

5. eval提示scaledsoftmax报错

   ```
   算子shape泛化性还有问题
   # 若遇到该报错
   搜索output文件夹生成的modeling_chatglm.py文件，
   self.scale_mask_softmax 设置为false
   ```

6. 遇到ImportError:/root/miniconda3/envs/codegeex/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block

   ```
   export LD_PRELOAD=/root/miniconda3/envs/codegeex/bin/../lib/libgomp.so.1
   ```

7. 微调时出现AttributeError或RuntimeError

module 'torch_npu' has no attribute 'npu_rotary_mul' 或

RuntimeError:Error!, The last dimension of input tensor shoule be within the range of [32,2048] and be divisible by32

```
修改modeling_chatglm.py文件:
USE_NPU_ROTARY=False
USE_SCALED_SOFTMAX=False
```

PS: 设置为True能提高性能

8. 如果cann不支持flash_attention

报错提示为module 'torch_npu' has no attribute 'npu_flash_attention'

```
修改modeling_chatglm.py文件:
USE_FLASH=False
```

​       PS: 设置为True能提高性能

9. 如果cann不支持npu_rms_norm

报错提示为module 'torch_npu' has no attribute 'npu_rms_norm'

```
修改modeling_chatglm.py文件:
USE_NPU_RMSNORM=False
```

​       PS: 设置为True能提高性能

