# VeRL for Pytorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

verl‌是一个集SFT（监督学习）与RL（强化学习）于一体的灵活大模型后训练框架。它特别适用于大型语言模型（LLM）的后训练阶段，旨在通过调整预训练模型的参数以适应新的任务或数据集。

- 参考实现：

  ```
  url=https://github.com/volcengine/verl
  commit_id=5b542d273cc792971eb66aca07494523be61c58c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/rl/
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
      <td> AscendHDK 25.1.RC1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.1.RC1 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.1.RC1 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.5.1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> 2.5.1 </td>
    </tr>
  </table>

- 安装vLLM和vLLM Ascend
  ```shell
  # 安装目录不能放在模型根目录下
  git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
  cd vllm
  pip install -r requirements-build.txt
  VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
  cd ..
  
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  
  # 对于VL模型，编译并安装vllm-ascend v0.7.3
  git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm-ascend.git
  cp -f 模型目录/vllm_ascend_need/qwen2_5_vl.py vllm-ascend/vllm_ascend/models/
  cp -f 模型目录/vllm_ascend_need/rotary_embedding.py vllm-ascend/vllm_ascend/ops/
  cd vllm-ascend
  export COMPILE_CUSTOM_KERNELS=1
  python setup.py install
  cd ..
  
  # 对于LLM模型，编译并安装vllm-ascend 特定commit id代码
  git clone https://github.com/vllm-project/vllm-ascend.git
  cd vllm-ascend
  git checkout edeadde387451ca982fe3717555c1841ee195718
  export COMPILE_CUSTOM_KERNELS=1
  python setup.py install
  cd ..
  ```

- 克隆transformers仓并切换到对应的commit id
  ```shell
  git clone --depth 1 https://github.com/huggingface/transformers.git
  cd transformers
  git fetch --depth 1 origin aa17cfb4d532239336d2f89e06f01d48387292a3
  git checkout aa17cfb4d532239336d2f89e06f01d48387292a3
  pip install -e .
  cd ..
  ```

- 对于VL模型，需要安装torchvision，克隆torchvision仓并切换到v0.20.1
  ```shell
  git clone -b v0.20.1 --depth 1  https://github.com/pytorch/vision.git
  cd vision
  python setup.py bdist_wheel
  # 安装`torchvision`前，需要先执行`pip uninstall torchvision`卸载原来的`torchvision`，如果环境中有`triton`，需要执行`pip uninstall triton`卸载
  pip install dist/*.whl
  cd ..
  ```

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -r requirements-npu.txt 
  pip install -e .
  ```


## 准备数据集

  VL模型使用geo3k数据集，在模型根目录下执行命令，下载并处理数据集，`--local_dir`为可选参数，不设置默认下载位置为`~/data/geo3k`。

  ```shell
  python examples/data_preprocess/geo3k.py --local_dir=xxx
  ```

  LLM模型使用gsm8k数据集，在模型根目录下执行命令，下载并处理数据集，`--local_dir`为可选参数，不设置默认下载位置为`~/data/gsm8k`。

  ```shell
  python examples/data_preprocess/gsm8k.py --local_dir=xxx
  ```

## 获取预训练模型

  用户自行下载`Qwen2.5-VL-7B-Instruct`、`Qwen2.5-VL-3B-Instruct`、`Qwen2.5-VL-32B-Instruct`、`Qwen2.5-7B-Instruct`和`Qwen2.5-32B-Instruct`模型。

# 开始训练

## 训练模型

使用`GRPO`算法进行训练。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```
   
2. 双机运行环境配置（单机环境请忽略）。

    1. 主从节点保证模型和数据集路径完全相同。

    2. 主节点执行以下命令启动ray集群：
       ```shell
       ray start --head
       ```

    3. 从节点执行以下命令加入ray集群：
       ```shell
       ray start --address='主节点ip:6379'
       ```

    4. 从节点执行以下命令确认双机已互联：
       ```shell
       ray status
       ```

3. 运行训练脚本。

   `Qwen2.5-VL-3B-Instruct`模型支持单机8卡训练。

   - 单机8卡训练

     ```shell
     bash test/train_qwen2_5_vl_3b_GRPO_full_8p.sh --data_path=xxx --model_path=xxx  # 8卡训练
     ```
     
   - 单机8卡性能
   
     ```shell
     bash test/train_qwen2_5_vl_3b_GRPO_performance_8p.sh --data_path=xxx --model_path=xxx   # 8卡性能
     ```
     
    `Qwen2.5-VL-7B-Instruct`模型支持单机16卡训练。

   - 单机16卡训练

     ```shell
     bash test/train_qwen2_5_vl_7b_GRPO_full_16p.sh --data_path=xxx --model_path=xxx   # 16卡训练
     ```
     
   - 单机16卡性能
   
     ```shell
     bash test/train_qwen2_5_vl_7b_GRPO_performance_16p.sh --data_path=xxx --model_path=xxx   # 16卡性能
     ```

    `Qwen2.5-VL-32B-Instruct`模型支持双机32卡训练。

   - 双机32卡训练

     ```shell
     # 主节点执行
     bash test/train_qwen2_5_vl_32b_GRPO_full_32p.sh --data_path=xxx --model_path=xxx   # 32卡训练
     ```
     
   - 双机32卡性能
   
     ```shell
     # 主节点执行
     bash test/train_qwen2_5_vl_32b_GRPO_performance_32p.sh --data_path=xxx --model_path=xxx   # 32卡性能
     ```

    `Qwen2.5-7B-Instruct`模型支持单机16卡训练。

   - 单机16卡训练

     ```shell
     bash test/train_qwen2_5_7b_instruct_GRPO_full_16p.sh --data_path=xxx --model_path=xxx   # 16卡训练
     ```
     
   - 单机16卡性能
   
     ```shell
     bash test/train_qwen2_5_7b_instruct_GRPO_performance_16p.sh --data_path=xxx --model_path=xxx   # 16卡性能
     ```

    `Qwen2.5-32B-Instruct`模型支持双机32卡训练。

   - 双机32卡训练

     ```shell
     # 主节点执行
     bash test/train_qwen2_5_32b_instruct_GRPO_full_32p.sh --data_path=xxx --model_path=xxx   # 32卡训练
     ```
     
   - 双机32卡性能
   
     ```shell
     # 主节点执行
     bash test/train_qwen2_5_32b_instruct_GRPO_performance_32p.sh --data_path=xxx --model_path=xxx   # 32卡性能
     ```
   
   训练完成后，训练日志保存在`test/output`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| MODEL                  | NAME                    | throughput | MAX Training TimeSteps |
|:-----------------------|:------------------------|:----------:|:----------------------:|
| Qwen2.5-VL-3B-Instruct | 8p-竞品A                  |  739.453   |           60           |
| Qwen2.5-VL-3B-Instruct | 8P Atlas 200T A2 Box16  |  349.013   |           60           |
| Qwen2.5-VL-7B-Instruct | 8p-竞品A                  |  568.452   |           60           |
| Qwen2.5-VL-7B-Instruct | 16P Atlas 200T A2 Box16 |  216.796   |           60           |
| Qwen2.5-7B-Instruct    | 8p-竞品A                  |  323.872   |           35           |
| Qwen2.5-7B-Instruct    | 16P Atlas 200T A2 Box16 |  190.617   |           35           |
| Qwen2.5-32B-Instruct   | 16p-竞品A                  |   79.022   |          105           |
| Qwen2.5-32B-Instruct   | 32P Atlas 200T A2 Box16 |   54.162   |          105           |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2025.05.12：首次发布。

## FAQ

- 如果在训练过程中遇到`RuntimeError: Gloo connectFullMesh failed`错误，请按照以下步骤操作：

  - 在主从节点分别执行以下命令获取节点ip对应的网口名称：

  ```shell
  ifconfig
  ```

  - 在主从节点分别设置以下环境变量：

  ```shell
  export GLOO_SOCKET_IFNAME=网口名称
  export HCCL_SOCKET_IFNAME=网口名称 
  ```
