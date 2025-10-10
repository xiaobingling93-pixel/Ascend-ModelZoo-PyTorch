# PPO for Pytorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

近端策略优化算法（Proximal Policy Optimization， PPO）是一种新型的Policy Gradient算法。为解决Policy Gradient算法中步长难以确定的问题，PPO提出了新的目标函数可以在多个训练步骤实现小批量的更新，是目前强化学习领域适用性最广的算法之一。


- 参考实现：

  ```
  url=https://github.com/nikhilbarhate99/PPO-PyTorch
  commit_id=6d05b5e3da80fcb9d3f4b10f6f9bc84a111d81e3
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

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  |  PyTorch 2.1  | Box2D==2.3.2 Box2D-kengz==2.3.3 gym==0.15.4 |
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -r requirements.txt
  pip install gym[box2d]==0.15.4
  ```


## 准备数据集

无。


## 获取预训练模型

无。

# 开始训练

## 训练模型

本文以BipedalWalker-v2场景为例，展示训练方法，其余场景需要根据场景替换启动脚本中的超参等配置。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     ```shell
     bash test/train_full_1p.sh  # 单卡训练
     ```
     
   - 单机单卡性能
   
     ```shell
     bash test/train_performance_1p.sh  # 单卡性能
     ```
   
   训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

| NAME         | FPS    | MAX Training TimeSteps | Average Reward |
|--------------| ------ | ---------------------- | -------------- |
| 1p-竞品V       | 585.37 | 3000000                 | 197.75         |
| 1p-NPU-Atlas 800T A2 | 284.02 | 3000000                | 240        |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 1p-竞品 | FP16 | 585.37 |
| 1p-Atlas 900 A2 PoDc | FP16 | 413.79 |
| 1p-Atlas 800T A2 | FP16 | 336.84 |

# 公网地址说明
无。

# 版本说明

## 变更

2023.08.20：首次发布。

## FAQ

无。
   
