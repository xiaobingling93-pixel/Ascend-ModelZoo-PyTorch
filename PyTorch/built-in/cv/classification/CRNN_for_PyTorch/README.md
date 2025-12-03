# CRNN for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

CRNN (Convolutional Recurrent Neural Network) 于2015年由华中科技大学的白翔老师团队提出，直至今日，仍旧是文本识别领域最常用也最有效的方法。

- 参考实现：

  ```
  url=https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec.git
  commit_id=90c83db3f06d364c4abd115825868641b95f6181
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```

# 准备训练环境

该模型为随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），您可以根据下面提供的安装指导选择匹配的CANN等软件下载使用。

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

- 三方库依赖如下表所示。

  **表 2**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1 | torchvision==0.16.0 |
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本

  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   模型训练以 MJSynth 数据集为训练集，IIIT 数据集为测试集。

   用户自行下载并解压 data_lmdb_release.zip，将其中的data_lmdb_release/training/MJ/MJ_train 文件夹 (重命名为 MJ_LMDB) 和 
   data_lmdb_release/evaluation/IIIT5k_3000 文件夹 (重命名为 IIIT5k_lmdb)上传至服务器的任意目录下，作为数据集目录。
   > 注意：若用户选择下载原始数据集，则需要将其转换为 lmdb 格式数据集，再根据上述步骤进行数据集上传。

   数据集目录结构参考如下所示：
   ```
   ├──服务器任意目录下
       ├──MJ_LMDB
             │──data.mdb
             │──lock.mdb
       ├──IIIT5K_lmdb
             │──data.mdb
             │──lock.mdb
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


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
     bash ./test/train_performance_1p.sh --data_path=数据集路径    # 单卡性能
     
     bash ./test/train_full_1p.sh --data_path=数据集路径           # 单卡精度 
     ```

   - 单机8卡训练

     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh --data_path=数据集路径    # 8卡性能
     
     bash ./test/train_full_8p.sh --data_path=数据集路径           # 8卡精度
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //训练集路径
   --epochs                            //重复训练次数
   --npu                               //npu训练卡id设置
   --max_step                          //设置最大迭代次数
   --stop_step                         //设置停止的迭代次数
   --profiling                         //设置profiling的方式
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

|  NAME  | Acc@1 |    FPS     | Epochs | AMP_Type | Torch_Version |
| :----: |:-----:|:----------:|:------:| :------: | :-----------: |
| 1p-NPU |   -   |  14758.65  |   1    |     O2     |     1.11      |
| 8p-NPU | 0.75  | 109015.73  |  100   |     O2     |     1.11      |
| 1p-NPU |   -   |  14078.58  |   1    |     O2     |      2.1      |
| 8p-NPU | 0.75  | 110879.797 |  100   |     O2     |      2.1      |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 8p-竞品 | FP16 | 113122.50 |
| 8p-Atlas 900 A2 PoDc | FP16 | 125304.48 |

# 版本说明
2022.02.17：更新readme，重新发布。

## FAQ
无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
