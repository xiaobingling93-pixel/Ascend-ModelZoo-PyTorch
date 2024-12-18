# Longformer for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

Longformer改进了Transformer传统的self-attention机制，是一种可高效处理长文本的模型。具体来说，每一个token只对固定窗口大小附近的token进行local attention（局部注意力）。并且Longformer针对具体任务，在原有local attention的基础上增加了一种global attention（全局注意力）。Longformer在两个字符级语言建模任务上都取得了SOTA的效果。用Longformer的attention方法继续预训练RoBERTa，训练得到的语言模型在多个长文档任务上进行fine-tune后，性能全面超越RoBERTa。


- 参考实现：

  ```
  url=https://github.com/huggingface/transformers.git
  commit_id=7378726df60b9cf399aacfe372fea629c1c4c7d3
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp/
  ```


# 准备训练环境

## 准备环境

- 推荐参考[配套资源文档](https://www.hiascend.com/developer/download/commercial)使用最新的配套版本。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 24.1.RC3 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 24.1.RC3 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.0.RC3 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v6.0.rc3 </td>
    </tr>
  </table>

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt
  ```

- 替换transformers库中相关文件
  
  将源码包根目录下transformers_modify文件夹中的各个文件分别替换到transformers库安装目录下的对应位置（基于**transformers 4.28.1**版本）：
  
  ```shell
  training_args.py -> transformers/training_args.py
  trainer.py -> transformers/trainer.py
  modeling_longformer.py -> transformers/models/longformer/modeling_longformer.py
  ```



## 获取预训练模型

联网情况下，预训练模型会自行下载。无网络情况下，用户需要自行下载预训练模型，并且将预训练模型所在本地路径传入训练脚本，参数名为--model_name。



## 准备数据集

1. 获取数据集。

   在源码包目录下新建文件夹corpus，用户自行获取数据集放至目录corpus下。

   Longformer数据集目录结构参考如下所示。

   ```
   ├── corpus
         ├──train_corpus.txt                     
         ├──test_corpus.txt             
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
  
2. 获取词表。
    在源码包目录下新建文件夹lf_token，用户自行获取词表，并放在目录lf_token下，目录结构如下。
   ```
   ├── lf_token
         ├──config.json
         ├──merges.txt
         ├──vocab.json
   ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh  # 单卡精度
     ```
     
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     bash ./test/train_performance_8p.sh  # 8卡性能
     ```
   
   训练完成后，权重文件保存在训练脚本指定的路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Perplexity |  FPS   | Epochs  | Torch_Version |
|:-:|:-:|:------:|:-:|:-:|
| 1p-竞品V | - | - | 3 | 1.11  |
| 8p-竞品V  | 3.0527 | 213.03 | 3  | 1.11  |
| 8p-NPU |  3.0603 | 157.22 |      3     |     1.11      |
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.11.20：首次发布。

## FAQ

1. 在conda环境下运行报错：ImportError: libblas.so.3: cannot open shared object file: No such file or directory
```
conda install openblas blas-devel

conda install -c conda-forge blas
```

2. 报错ImportError:xxx/python3.7/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
```
find / -name "libgomp-d22c30c5.so.1.0.0"

export LD_PRELOAD=$LD_PRELOAD:/path/to/your/conda_env/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
```

