# 当前模型脚本已不随版本演进，如使用此模型可跳转至该[地址](https://gitee.com/ascend/MindSpeed-LLM)

# ChatGLM3-6B for PyTorch


## 简介

**ChatGLM3** 是智谱AI和清华大学 KEG 实验室联合发布的对话预训练模型。ChatGLM3-6B 是 ChatGLM3
系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：

1. **更强大的基础模型：** ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base
   采用了更多样的训练数据、更充分的训练步数和更合理的训练策略。在语义、数学、推理、代码、知识等不同角度的数据集上测评显示，*
   *ChatGLM3-6B-Base 具有在 10B 以下的基础模型中最强的性能**。
2. **更完整的功能支持：** ChatGLM3-6B 采用了全新设计的Prompt格式，除正常的多轮对话外。同时原生支持工具调用（Function Call）、代码执行（Code Interpreter）和Agent 任务等复杂场景。
3. **更全面的开源序列：** 除了对话模型ChatGLM3-6B外，还开源了基础模型ChatGLM3-6B-Base
   、长文本对话模型ChatGLM3-6B-32K和进一步强化了对于长文本理解能力的ChatGLM3-6B-128K。以上所有权重对学术研究**完全开放**
   ，在填写 [问卷](https://open.bigmodel.cn/mla/form) 进行登记后**亦允许免费商业使用**。。
- 参考实现 ：
  ```
  url=https://github.com/THUDM/ChatGLM3
  commitID=d0be06cd278eb58541b971f69f0544b75613ebdd
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```

## 准备训练环境

### 准备环境

#### 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|软件类型	|支持版本|
|-----|-----|
|FrameworkPTAdapter	|6.0.RC2|
|CANN	|8.0.RC2|
|昇腾NPU固件	|24.1.RC2|
|昇腾NPU驱动	|24.1.RC2|
#### 安装模型环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1 |transformers == 4.39.2; deepspeed == 0.14.2 |
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```python
  pip install -r requirements.txt
  ```

### 准备数据集

  用户可以自行下载AdvertiseGen数据集，并将其放在`scripts`路径下，该文件夹内容包括：
```
├── AdvertiseGen
|      ├──train.json
|      ├──dev.json
├── configs
│     ├── ds_zero_3.json
│     └── sft.yaml
├── finetune_hf.py
├── run_train_8p_glm.sh
├── run_train_8p_glm_32k.sh
├── run_train_8p_glm_32k_no_shuffle.sh
└── run_train_8p_glm_no_shuffle.sh
```

### 准备预训练权重
#### ChatGLM3-6B
  用户可以自行下载ChatGLM3-6B预训练权重和配置文件，然后将这些文件放在 "model"文件夹中，**不要覆盖 `modeling_chatglm.py`文件**。
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

#### ChatGLM3-6B-32K

  用户可以自行下载ChatGLM3-6B-32K预训练权重和配置文件，然后将这些文件放在 "model-32K"文件夹中，**不要覆盖 `modeling_chatglm.py`文件**。
`model-32K`文件夹内容如下：
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
## 开始训练

### ChatGLM3-6B
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```
2. 启动全参数finetune训练

    启动训练环境
    ```shell
    # 调整以下`set_env`脚本为ascend-toolkit的安装路径
    source set_env.sh
    ```
    启动8卡微调，打乱数据

    ```shell
    bash scripts/run_train_8p_glm.sh
    ```
    启动8卡微调，不打乱数据

    ```shell
    bash scripts/run_train_8p_glm_no_shuffle.sh
    ```
### ChatGLM3-6B-32K
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```
2. 启动全参数finetune训练

    启动训练环境
    ```shell
    # 调整以下`set_env`脚本为ascend-toolkit的安装路径
    source set_env.sh
    ```
    启动8卡微调，打乱数据

    ```shell
    bash scripts/run_train_8p_glm_32k.sh
    ```
    启动8卡微调，不打乱数据

    ```shell
    bash scripts/run_train_8p_glm_32k_no_shuffle.sh
    ```

    模型训练脚本参数说明如下
    ```
    data_dir                                        //数据集文件夹路径
    model_dir                                       //模型文件夹路径
    config_file                                     //配置文件路径
    shuffle                                         //训练时是否打乱数据集
    auto_resume_from_checkpoint                     //是否自动从上一次训练中最后一个checkpoint继续训练
    ```

    如需要保存日志文件，请使用以下命令，并将运行脚本及日志文件名更改为自己对应的文件名
    ```bash
    nohup bash /path/to/your/sh/scripts >/path/to/your/log/file 2>&1 &
    ```

### 训练结果展示

**表 3**  训练结果展示表
| 芯片    | 卡数 | 模型       | Iterations | Global Batch Size | Train Samples per Second 
| --------- |---| ----------- | ---------------- | ----------------------------- | ---------------------------- |
| Atlas A2 |8p| ChatGLM3-6B     | 2000  | 16 |13.781  |
| GPU      |8p| ChatGLM3-6B     | 2000  | 16 |15.094  |
| Atlas A2 |8p| ChatGLM3-6B-32K | 2000  | 16 | 11.819 |
| GPU      |8p| ChatGLM3-6B-32K | 2000  | 16 |12.088  |




## 公网地址说明

代码涉及公网地址参考 public_address_statement.md

## 变更说明
2024.05.29：首次发布

## FAQ
暂无。