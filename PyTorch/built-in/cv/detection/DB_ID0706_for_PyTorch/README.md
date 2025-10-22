# DBNet for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

DB(Differentiable Binarization)是一种使用可微分二值图来实时文字检测的方法，
和之前方法的不同主要是不再使用硬阈值去得到二值图，而是用软阈值得到一个近似二值图，
并且这个软阈值采用sigmoid函数，使阈值图和近似二值图都变得可学习。

- 参考实现：

  ```
  url=https://github.com/MhLiao/DB
  commit_id=4ac194d0357fd102ac871e37986cb8027ecf094e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
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

- 安装geos，可按照环境选择以下方式：

  1. ubuntu系统：

     ```
     sudo apt-get install libgeos-dev
     ```

  2. euler系统：

     ```
     sudo yum install geos-devel
     ```

  3. 源码安装：

     ```
     wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
     bunzip2 geos-3.8.1.tar.bz2
     tar xvf geos-3.8.1.tar
     cd geos-3.8.1
     ./configure && make && make install
     ```

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.11_requirements.txt  # PyTorch1.11版本

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。
    
    请用户自行下载 `icdar2015` 数据集，解压放在任意文件夹 `datasets`下，数据集目录结构参考如下所示。

    ```
    |--datasets
       |--icdar2015
    ```

    > **说明：** 
    >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

请用户自行获取预训练模型，将获取的 `MLT-Pretrain-Resnet50` 预训练模型，放至在源码包根目录下新建的 `path-to-model-directory` 目录下。


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
      1.安装环境，确认预训练模型放置路径，若该路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
      2.开始训练
        bash ./test/train_full_1p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 单卡精度
        bash ./test/train_performance_1p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 单卡性能   
      ```
      **注意**：如果发现打屏日志有报checkpoint not found的warning，请再次检查预训练模型MLT-Pretrain-Resnet50的配置，以免影响精度。

   - 单机8卡训练

     启动8卡训练。

      ```
      1.安装环境，确认预训练模型放置路径，若该路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
      2.开始训练
        bash ./test/train_full_8p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 8卡精度
        bash ./test/train_performance_8p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 8卡性能    
      ```
   - 在线推理

     启动在线推理

      ```
        bash ./test/eval.sh --data_path=${datasets} --resume=${resume}    #在线推理
      ```
    

   --data_path参数填写数据集路径，需写到数据集的一级目录，--resume参数填写模型权重

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                          //数据集路径
   --addr                              //主机地址
   --num_workers                       //加载数据进程数      
   --epochs                            //重复训练次数
   --batch_size                        //训练批次大小，默认：240
   --lr                                //初始学习率
   --amp                               //是否使用混合精度
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Precision |   FPS   | Epochs |  AMP_Type | Torch_Version |
|:----:|:---------:|:-------:|:------:| :-------: |:---:|
| 1P-竞品V |     -     | - |   1    |       - | 1.5 |
| 8P-竞品V |     -     | - |  1200  |       - | 1.5 |
| 1p-NPU |     -     | 30.528  |   1    |    O2     |     1.11      |
| 8p-NPU |   0.907   | 210.123 |  1200  |    O2     |     1.11      |
| 1p-NPU |     -     | 29.926  |   1    |    O2     |      2.1      |
| 8p-NPU |   0.907   | 205.123 |  1200  |    O2     |      2.1      |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 8p-竞品 | FP16 | 179.36 |
| 8p-Atlas 900 A2 PoDc | FP16 | 203.4 |

# 版本说明

## 变更

2022.12.23：Readme 整改。

## FAQ
#### DataLoader work is killed by signal: Segmentation fault.
可以参考社区[issue](https://github.com/pytorch/pytorch/issues/54752)，调整num_workers参数。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
