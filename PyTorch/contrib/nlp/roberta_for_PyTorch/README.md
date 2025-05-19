# RoBERTa for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

RoBERTa 模型更多的是基于 BERT 的一种改进版本。是 BERT 在多个层面上的重大改进。
RoBERTa 在模型规模、算力和数据上，都比 BERT 有一定的提升。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq/tree/main/examples/roberta 
  commit_id=d871f6169f8185837d1c11fb28da56abfd83841c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/nlp
  ```


# 准备训练环境

该模型为随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），您可以根据下面提供的安装指导选择匹配的CANN等软件下载使用。

## 准备环境

- 推荐使用最新的版本准备训练环境。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 25.0.RC1.1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.0.RC1.1 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.1.RC1 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v7.0.0-pytorch2.1.0 </td>
    </tr>
  </table>
  
- 安装依赖：

  ```
  pip install -r 2.0_requirements.txt # PyTorch2.0及以上版本
  python3 setup.py build_ext --inplace
  ```
  > **说明：** 
  >安装requirements.txt中的依赖只需执行一条对应的PyTorch版本依赖安装命令。


## 训练准备

1. 获取数据集。

   下载 `SST-2` 数据集，请参考 `examples/roberta/preprocess_GLUE_tasks.sh` 。

   `SST-2` 数据集目录结构参考如下所示。

   ```
   ├── SST-2
         ├──input0
              ├──dict.txt
              │──preprocess.log
              │──test.bin
              │——test.idx   
              ├──train.bin
              │──train.idx
              │──valid.bin
              │——valid.idx                    
         ├──label
              ├──dict.txt
              │──preprocess.log 
              ├──train.bin
              │──train.idx
              │──valid.bin
              │——valid.idx              
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 获取预训练模型

    下载预训练模型 `RoBERTa.base` , 解压至源码包路径下：“./pre_train_model/RoBERTa.base/model.pt”。


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
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=real_data_path # 8卡性能 
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //数据集路径
   --restore-file                           //权重文件保存路径
   --max-tokens                             //最大token值
   --num-classes                            //分类数      
   --max-epoch                              //重复训练次数
   --batch-size                             //训练批次大小
   --lr                                     //初始学习率，默认：0.01
   --use-apex                               //使用混合精度
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表2**  训练结果展示表

|  NAME  | Acc@1 |   FPS   | Epochs | AMP_Type | Torch_Version |
|:------:| :---: |:-------:|:------:|:--------:|:-------------:|
| 1p-竞品V | 0.927     |  397 |   1    | - | 1.5 |
| 8p-竞品V | 0.943 | 2997 |   10   | - | 1.5 |
| 1p-NPU | 0.938 | 902.265 |   1    |   O2     |     1.11      |
| 8p-NPU | 0.969 | 7111.11 |   10   |   O2    |     1.11      |
| 1p-NPU | 0.938 | 879.05 |   1    |    O2     |      2.1      |
| 8p-NPU | 0.969 | 7078.64 |   10   |    O2     |      2.1      |

说明：上表为历史数据，仅供参考。2024年12月31日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 8p-竞品 | FP16 | 7309.36 |
| 8p-Atlas 900 A2 PoDc | FP16 | 8084.65 |

# 版本说明

## 变更

2022.08.24：首次发布

## FAQ

无。











