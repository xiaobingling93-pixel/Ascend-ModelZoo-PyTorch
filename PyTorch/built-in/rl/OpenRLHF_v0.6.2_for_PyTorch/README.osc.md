# 目前本仓已停止维护，如有使用请严格遵守README版本配套
# OpenRLHF

# 概述

OpenRLHF是一个基于Ray、DeepSpeed和HF Transformers构建的高性能RLHF框架。

# 支持算法

| 算法      | PyThon版本 | PyTorch版本 | 训练指南                  |
| :-------- | ---------- | ----------- | ------------------------- |
| PPO、GRPO | 3.10       | 2.5.1       | [训练指南-1](#训练指南-1) |
| KTO、RM、PRM | 3.11       | 2.6.0       | [训练指南-2](#训练指南-2) |

# 训练指南-1

### 环境准备

* 推荐使用最新的版本准备训练环境。

  **表 1**  版本配套表
  
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 25.0.RC1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.0.RC1 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.1.RC1 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.5.1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/index/index.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> 2.5.1 </td>
    </tr>
  </table>

* 安装vLLM和vLLM Ascend

  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  
  git clone -b v0.7.3 https://github.com/vllm-project/vllm.git
  cd vllm
  pip install -r requirements-build.txt
  VLLM_TARGET_DEVICE=empty pip install .
  cd ..
  
  git clone -b v0.7.3 https://github.com/vllm-project/vllm-ascend.git
  cd vllm-ascend
  sed -i '14,15s/^/#/' pyproject.toml # 已经在前面步骤安装torch和torch-npu，将其注释
  pip install -e .
  cd ..
  ```

* 安装OpenRLHF

  ```shell
  # 在当前OpenRLHF_v0.6.2_for_PyTorch文件夹路径下执行
  TARGET_DEVICE=NPU pip install -e .
  ```

* 安装transformers

  * 克隆transformers仓

    ```shell
    git clone -b v4.51.0 https://github.com/huggingface/transformers.git
    ```
  
  * 使用transformers_need下的模型文件替换transformers的原始模型实现，使能融合算子
  
    ```shell
    cp -f transformers_need/modeling_qwen2.py transformers/src/transformers/models/qwen2/modeling_qwen2.py
    cp -f transformers_need/modeling_llama.py transformers/src/transformers/models/llama/modeling_llama.py
    cp -f transformers_need/npu_flash_attention.py transformers/src/transformers/integrations/npu_flash_attention.py
    ```

  * 进入transformers目录，执行编译安装命令

    ```shell
    cd transformers
    pip install -e .
    cd ..
    ```

* 卸载 torchvision：解决循环依赖（circular import）报错

  ```shell
  pip uninstall torchvision
  ```

### 数据集准备

下载math_level3to5_data_processed_with_qwen_prompt数据集。新建data文件夹，将数据集对应的json文件放入data文件夹中。

### 模型权重准备

下载Qwen2.5-7B-Instruct模型权重。新建models文件夹，将模型权重放入models文件夹中。

### RM脚本准备

下载 `orm_server.py` 文件，放入 `openrlhf/cli` 文件夹下。

### 开始训练

#### PPO算法

运行训练脚本。该算法支持单机16卡训练。

```shell
# 16卡训练
bash test/train_ppo_full_16p.sh --model_path=./models/xxx --dataset_path=./data/xxx
# 16卡性能
bash test/train_ppo_performance_16p.sh --model_path=./models/xxx --dataset_path=./data/xxx
```

#### GRPO算法

运行训练脚本。该算法支持单机16卡训练。

```shell
# 16卡训练
bash test/train_grpo_full_16p.sh --model_path=./models/xxx --dataset_path=./data/xxx
# 16卡性能
bash test/train_grpo_performance_16p.sh --model_path=./models/xxx --dataset_path=./data/xxx
```

#### 训练结果展示

| 模型                | 算法 | 芯片          | 卡数 | 单步时间（小时） |
| ------------------- | ---- | ------------- | ---- | ---------------- |
| Qwen2.5-7B-Instruct | PPO  | Atlas 200T A2 Box16 | 16p  | 2.06 |
| Qwen2.5-7B-Instruct | PPO  | 竞品A         | 16p  | 1.85             |
| Qwen2.5-7B-Instruct | GRPO | Atlas 200T A2 Box16 | 16p  | 2.04 |
| Qwen2.5-7B-Instruct | GRPO | 竞品A         | 16p  | 1.85             |

# 训练指南-2

### 环境准备

* 推荐使用最新的版本准备训练环境。

* 安装OpenRLHF

  ```shell
  # 在当前OpenRLHF_v0.6.2_for_PyTorch文件夹路径下执行
  TARGET_DEVICE=NPU_2_6 pip install -e .
  ```

* 安装transformers

  * 克隆transformers仓并切换到对应的commit id

    ```shell
    git clone -b v4.51.0 https://github.com/huggingface/transformers.git
    ```
  
  * 使用transformers_need下的模型文件替换transformers的原始模型实现，使能融合算子
  
    ```shell
    cp -f transformers_need/modeling_qwen2.py transformers/src/transformers/models/qwen2/modeling_qwen2.py
    cp -f transformers_need/modeling_llama.py transformers/src/transformers/models/llama/modeling_llama.py
    cp -f transformers_need/npu_flash_attention.py transformers/src/transformers/integrations/npu_flash_attention.py
    ```

  * 进入transformers目录，执行编译安装命令

    ```shell
    cd transformers
    pip install -e .
    cd ..
    ```

### 数据集准备

  在当前OpenRLHF_v0.6.2_for_PyTorch文件夹下新建data文件夹。

  - KTO

    下载ultrafeedback-unpaired-preferences数据集，将数据集放入data文件夹中。
  
  - RM

    下载preference_dataset_mixture2_and_safe_pku数据集，将数据集放入data文件夹中。
  
  - PRM

    下载Math-Shepherd数据集，将数据集放入data文件夹中。

### 模型权重准备

  在当前OpenRLHF_v0.6.2_for_PyTorch文件夹下新建models文件夹。

  - KTO、RM

    下载Llama-3-8b-sft-mixture模型权重，将模型权重放入models文件夹中。

  - PRM

    下载Mistral-7B-v0.1模型权重，将模型权重放入models文件夹中。

### 开始训练

#### KTO算法

运行训练脚本。

```shell
# 8卡训练
bash test/train_kto_full_8p.sh --pretrain_path=./models/Llama-3-8b-sft-mixture --dataset_path=./data/ultrafeedback-unpaired-preferences
# 8卡性能
bash test/train_kto_performance_8p.sh --pretrain_path=./models/Llama-3-8b-sft-mixture --dataset_path=./data/ultrafeedback-unpaired-preferences
```

#### RM算法

运行训练脚本。

```shell
# 8卡训练
bash test/train_rm_full_8p.sh --pretrain_path=./models/Llama-3-8b-sft-mixture --dataset_path=./data/preference_dataset_mixture2_and_safe_pku
# 8卡性能
bash test/train_rm_performance_8p.sh --pretrain_path=./models/Llama-3-8b-sft-mixture --dataset_path=./data/preference_dataset_mixture2_and_safe_pku
```

#### PRM算法

运行训练脚本。

```shell
# 8卡训练
bash test/train_prm_full_8p.sh --pretrain_path=./models/Mistral-7B-v0.1 --dataset_path=./data/Math-Shepherd/data
# 8卡性能
bash test/train_prm_performance_8p.sh --pretrain_path=./models/Mistral-7B-v0.1 --dataset_path=./data/Math-Shepherd/data
```

#### 训练结果展示

| 模型                | 算法 | 芯片          | 卡数 | 单步时间（秒） |
| ------------------- | ---- | ------------- | ---- | ---------------- |
| Llama-3-8b-sft-mixture | KTO | Atlas 900 A2 PODc | 8p  | 1.54 |
| Llama-3-8b-sft-mixture | KTO | 竞品A             | 8p  | 0.96 |
| Llama-3-8b-sft-mixture | RM  | Atlas 900 A2 PODc | 8p  | 1.49 |
| Llama-3-8b-sft-mixture | RM  | 竞品A             | 8p  | 1.14 |
| Mistral-7B-v0.1        | PRM | Atlas 900 A2 PODc | 8p  | 1.71 |
| Mistral-7B-v0.1        | PRM | 竞品A             | 8p  | 1.50 |

# 版本说明

## 变更

2025.5.27：首次发布

# FAQ

* 使用--adam_offload参数可能存在长时间卡顿的情况，解决方法是删除torch_extensions的缓存文件，参考[issue](https://github.com/deepspeedai/DeepSpeed/issues/2816#issuecomment-1450095538)。
* 在 Atlas 200T A2 Box16 机器中，如果使用了跨平面的卡，需要使能环境变量 `export HCCL_INTRA_ROCE_ENABLE=1`，使用RoCE环路进行多卡间的通信。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
