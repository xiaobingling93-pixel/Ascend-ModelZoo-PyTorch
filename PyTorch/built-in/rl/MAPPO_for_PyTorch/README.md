# MAPPO for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

多智能体近端策略优化算法（Multi-Agent Proximal Policy Optimization， MAPPO）是一种新型的Policy Gradient算法。基于现有的近端策略优化算法（Proximal Policy Optimization， PPO），在不修改算法架构的基础上，通过调整超参数，在多智能体环境中达到与大多数off-policy算法相当的性能。


- 参考实现：

  ```
  url=https://github.com/marlbenchmark/on-policy
  commit_id=b21e0f743bd4516086825318452bb6927a33538d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/rl/
  ```

# 准备训练环境

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

  **表 2**  三方库依赖表

  | Torch_Version      |                           三方库依赖版本                            |
  | :--------: |:------------------------------------------------------------:|
  |  PyTorch 2.1  |  absl-py==1.4.0; gym==0.17.2; protobuf==3.20.0; wandb==0.10.5  |
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -r requirements.txt  
  pip install -e .
  ```


## 准备数据集

无。


## 获取预训练模型

无。

# 开始训练

## 训练模型

本文以MPE Comm场景为例，展示训练方法，其余场景需要根据场景替换启动脚本。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练。

   - 单机单卡训练

     ```shell
     bash test/train_full_1p.sh  # 单卡训练
     ```
     
   - 单机单卡性能
   
     ```shell
     bash test/train_performance_1p.sh  # 单卡性能
     ```
   
   训练完成后，权重文件保存在`onpolicy/scripts/results`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

| NAME      | FPS  | MAX Training TimeSteps | Average Reward |
|-----------|------|------------------------|----------------|
| 1p-竞品V   | 1789 | 2000000                | -15.9          |
| 1p-NPU    | 885  | 2000000                | -15.9          |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 1p-竞品 | FP16 | 1789 |
| 1p-Atlas 900 A2 PoDc | FP16 | 1791.33 |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.09.05：首次发布。

## FAQ

无。
   
