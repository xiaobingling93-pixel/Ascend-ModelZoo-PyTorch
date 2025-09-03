# Albert for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

Albert是自然语言处理模型，基于Bert模型修改得到。相比于Bert模型，Albert的参数量缩小了10倍，减小了模型大小，加快了训练速度。在相同的训练时间下，Albert模型的精度高于Bert模型。

- 参考实现：

  ```
  url=https://github.com/lonePatient/albert_pytorch 
  commit_id=46de9ec6b54f4901f78cf8c19696a16ad4f04dbc
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
  
- 安装依赖。

  在模型根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```



## 准备数据集

1. 获取数据集。

   用户自行下载 `SST-2` 和 `STS-B` 数据集，在模型根目录下创建 `dataset` 目录，并放入数据集。

   数据集目录结构参考如下所示。

   ```
   ├── dataset
         ├──SST-2
              ├──original
              │──dev.tsv
              │──test.tsv
              │──train.tsv   
              |  ...                     
         ├──STS-B  
              ├──original
              │──dev.tsv
              │──test.tsv
              │──train.tsv
              │   ...              
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 下载预训练模型
下载 `albert_base_v2` 预训练模型，在模型根目录下创建 `prev_trained_model` 目录，并将预训练模型放置在该目录下。

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
     bash ./test/train_full_1p.sh --data_path=real_data_path         #单卡精度
     bash ./test/train_performance_1p.sh --data_path=real_data_path  #单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path         #8卡精度
     bash ./test/train_performance_8p.sh --data_path=real_data_path  #8卡性能 
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=real_data_path  #8卡评测
     ```
   --data\_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_dir                           //数据集路径
   --model_type                         //模型类型
   --task_name                          //任务名称
   --output_dir                         //输出保存路径
   --do_train                           //是否训练
   --do_eval                            //是否验证
   --num_train_epochs                   //重复训练次数
   --batch-size                         //训练批次大小
   --learning_rate                      //初始学习率
   --fp16                               //是否使用混合精度
   --fp16_opt_level                     //混合精度的level
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表2**  训练结果展示表

|   NAME   | Acc@1 |   FPS   | Epochs | AMP_Type | Torch_Version |
| :------: | :---: |:-------:|:------:| :------: |:-------------:|
| 1p-竞品V | 0.927 | 517  |   2    |    O1     |      1.5      |
| 8p-竞品V | 0.914 | 3327 |   7    |    O1     |      1.5      |
| 1p-NPU |  0.932 | 1042.69 |   2    |    O2     |     1.11      |
| 8p-NPU | 0.927 | 6479.72 |   7    |     O2     |     1.11      |
| 1p-NPU |  0.932 | 1025.36 |   2    |     O2     |      2.1      |
| 8p-NPU | 0.927 | 6394.05 |   7    |     O2     |      2.1      |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------: |:-------:|:------:|
| 8p-竞品 | FP16 | 6394.05 |
| 8p-Atlas 900 A2 PoDc | FP16 | 9186.47 |


# 版本说明

## 变更

2022.08.24：首次发布。

## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md   
