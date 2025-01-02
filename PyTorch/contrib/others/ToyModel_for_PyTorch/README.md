# ToyModel for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [版本说明](#版本说明)



# 概述

## 简述

ToyModel是一个旨在指导用户将模型从GPU或其他第三方框架迁移适配到NPU的示例模型。用户可以参考《PyTorch训练模型迁移调优指南》中的[迁移适配章节](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/PT_LMTMOG_0011.html)，进行具体操作。该文档详细介绍了如何通过修改模型代码等适配步骤，使原本为GPU或其他第三方平台设计的深度学习模型适应NPU的架构和编程环境，确保模型能在NPU上顺利运行。结合ToyModel的示例，文档为用户提供了端到端迁移适配操作指导。


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

  请参考《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

启动脚本后会自动下载MNIST数据集，无需自行准备。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡和多机训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/run_1p.sh  # 单卡不开启混合精度
     
     bash ./test/run_1p_with_amp.sh  # 单卡开启混合精度
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/run_8p.sh  # 单机8卡不开启混合精度
     
     bash ./test/run_8p_with_amp.sh  # 单机8卡开启混合精度
     ```

    - 双机16卡训练

      启动16卡训练。

      ```
      bash ./test/run_16p.sh  # 双机16卡不开启混合精度
      
      bash ./test/run_16p_with_amp.sh  # 双机16卡开启混合精度
      ```
      启动双机脚本时，需要根据实际情况修改对应的MASTER_ADDR和NODE_RANK值。

# 版本说明

## 变更

2024.12.31：首次发布

## FAQ

无。
