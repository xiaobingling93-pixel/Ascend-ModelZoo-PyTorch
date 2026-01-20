# TorchTitan昇腾插件

## 概述

### 简介

本项目开发了名为**torchtitan_npu_patch**的TorchTitan昇腾插件，使昇腾NPU可以适配TorchTitan框架，为使用TorchTitan框架的开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 准备训练环境

### 版本配套

- 硬件：昇腾服务器，已验证Atlas 800T A2（Kunpeng/X86 CPU）
- 系统：Linux，已验证openEuler 22.03 (LTS-SP4)
- Python：已验证3.10、3.11

已验证软件栈：

| CANN| torch | torch_npu | torchtitan | triton-ascend |
| --- | --- | --- | --- | --- |
| 8.5.0.B120 | 2.9.0 | 2.9.0.post1.dev20260108 | 0.2.0 | 3.4.0.dev2026010713 |

### 昇腾CANN安装配置

确保昇腾NPU固件和驱动程序是否已正确安装，运行以下命令可确认NPU信息、健康状态和使用状态：

```bash
npu-smi info
# output e.g.
# +------------------------------------------------------------------------------------------------+
# | npu-smi 25.0.rc1.1               Version: 25.0.rc1.1                                           |
# +---------------------------+---------------+----------------------------------------------------+
# | NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
# | Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
# +===========================+===============+====================================================+
# | 0     DEVICE_NAME         | OK            | 181.4       47                0    / 0             |
# | 0                         | 0000:C1:00.0  | 49          0    / 0          44421/ 65536         |
# ...
```

执行以下命令确认CANN是否安装以及对应版本，以下示例为默认安装路径，实际路径可能不同：

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
# output e.g.
# package_name=Ascend-cann-toolkit
# version: 8.5.0
# ...
```

下载和安装详情参考[Ascend环境搭建指南](https://ascend.github.io/docs/sources/ascend/quick_install.html)或[昇腾社区](https://www.hiascend.com/developer/download/)。

确认昇腾固件、驱动、CANN安装配置完成后，可直接跳到快速开始章节快速安装余下软件，也可逐步按照下述章节中各组件安装指导安装余下软件。

### 安装PyTorch和torch_npu

参考[torch_npu环境部署](https://gitcode.com/Ascend/pytorch?tab=md#%E7%8E%AF%E5%A2%83%E9%83%A8%E7%BD%B2)，了解昇腾版本配套关系，安装PyTorch、torch_npu。

### 安装triton-ascend

安装triton-ascend：

```bash
pip install triton-ascend
```

### 安装TorchTitan

参考[TorchTitan Installation](https://github.com/pytorch/torchtitan/tree/v0.2.0?tab=readme-ov-file#installation)安装TorchTitan。
或者直接下载[TorcTitan-v0.2.0源码](https://github.com/pytorch/torchtitan/tree/v0.2.0)，配置`PYTHONPATH=torchtitan根目录`。

### 快速开始

安装PyTorch、torch_npu、triton-ascend等，参考：

```bash
# After ascend firmware/driver and CANN are installed correctly
# After python is installed or conda python env is activated

# Config pip mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# For torch-npu dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

# Install pytorch
pip install torch==2.9.0

# Install torch-npu prerequest
pip install pyyaml
pip install setuptools
pip install psutil
pip install scipy
pip install decorator
pip install numpy==1.2x
# Install torch-npu
pip install torch_npu==2.9.0.post1.dev20260108 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

# Install triton-ascend
pip install triton-ascend

# Download torchtitan
git clone --branch v0.2.0 https://github.com/pytorch/torchtitan.git
# Install torchtitan prerequest
pip install -r torchtitan/requirements.txt
# Add torchtitan to PYTHONPATH
export PYTHONPATH=`pwd`/torchtitan/:$PYTHONPATH
```

## 补丁文件说明

将`torchtitan_npu_patch.py`添加至torchtitan根目录`torchtitan/train.py`同级目录下，在`train.py`开头添加
注：补丁文件`torchtitan_npu_patch.py`与本`README.md`在同一目录下

```python
from . import torchtitan_npu_patch
# before import torch
```

补丁文件主要补充了兼容性、亲和性能优化等修改。

## 运行

### 预训练数据集下载

可使用测试数据集`c4_test`(无需下载)，模型配置文件中已默认指定，无需修改。如果使用C4数据集，请下载并在对应模型配置文件中修改

```toml
# set or comment dataset and add dataset_path
dataset = "c4"
dataset_path = "/PATH/TO/c4"

# or save c4 dataset in default directory "allenai/c4" in torchtian source root, and set:
dataset = "c4"
```

### DeepSeekV3配置

参考[torchtitan社区Deepseek_V3模型](https://github.com/pytorch/torchtitan/tree/v0.2.0/torchtitan/models/deepseek_v3)。

Deepseek_V3配置文件路径`./torchtitan/models/deepseek_v3/train_configs/`。

#### tokenizer下载

运行以下命令可下载 Hugging Face 上的指定资源（如 tokenizer）：

```bash
cd torchtitan source root

python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer
```

`<hf_repo_name>`可选：
- 对于deepseek_v3_16b模型，Hugging Face 仓库名称为`deepseek-ai/deepseek-moe-16b-base`；

#### 训练配置

deepseek_v3_16b FSDP2训练配置为

```toml
[training]
local_batch_size = 4 #Atlas 800T A2单机8卡开箱训练配置会OOM，需减小batch_size与seq_len拉起实验
seq_len = 2048

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
fsdp_reshard_after_forward = "default" # default / never / always
tensor_parallel_degree = 1
enable_async_tensor_parallel = false
pipeline_parallel_degree = 1
pipeline_parallel_schedule = "Interleaved1F1B"
expert_parallel_degree = 1
expert_tensor_parallel_degree = 1

[compile]
enable=false #目前版本尚不支持compile选项，相应配置应为false
components = ["model", "loss"]
```

deepseek_v3_16b FSDP2+EP8训练配置为

```toml
[training]
local_batch_size = 4 #Atlas 800T A2单机8卡开箱训练配置会OOM，需减小batch_size与seq_len拉起实验
seq_len = 2048

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
fsdp_reshard_after_forward = "default" # default / never / always
tensor_parallel_degree = 1
enable_async_tensor_parallel = false
pipeline_parallel_degree = 1
pipeline_parallel_schedule = "Interleaved1F1B"
expert_parallel_degree = 8
expert_tensor_parallel_degree = 1

[compile]
enable=false #目前版本尚不支持compile选项，相应配置应为false
components = ["model", "loss"]
```

deepseek_v3_16b HSDP2+TP2+EP4训练配置为

```toml
[training]
local_batch_size = 4 #Atlas 800T A2单机8卡开箱训练配置会OOM，需减小batch_size与seq_len拉起实验
seq_len = 2048

[parallelism]
data_parallel_replicate_degree = 2
data_parallel_shard_degree = -1
fsdp_reshard_after_forward = "default" # default / never / always
tensor_parallel_degree = 2
enable_async_tensor_parallel = false
pipeline_parallel_degree = 1
pipeline_parallel_schedule = "Interleaved1F1B"
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1

[compile]
enable=false #目前版本尚不支持compile选项，相应配置应为false
components = ["model", "loss"]
```

deepseek_v3_16b SAC=none, FSDP2+TP2+EP4+PP2-ZBVZeroBubble训练配置为

```toml
[training]
local_batch_size = 4 #Atlas 800T A2单机8卡开箱训练配置会OOM，需减小batch_size与seq_len拉起实验
seq_len = 2048

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
fsdp_reshard_after_forward = "default" # default / never / always
tensor_parallel_degree = 2
enable_async_tensor_parallel = false
pipeline_parallel_degree = 2
pipeline_parallel_schedule = "ZBVZeroBubble" #该项配置为ZBVZeroBubble后需将SAC设置为none,其他配置对SAC无影响
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1

[activation_checkpoint]
mode = "none"  #["none", "selective", "full"] 上述配置需将该项配置为none，否则会报错，GPU存在同样现象
selective_ac_option = 'op'  # 'int' = ac every positive int layer or 'op', ac based on ops policy

[compile]
enable=false #目前版本尚不支持compile选项，相应配置应为false
components = ["model", "loss"]
```

使用PP并行只会在模型划分好的最后一个stage输出loss，以上述PP=2为例，在双节点上执行上述并行策略时，不同的模型stage会被放置在不同的节点上，因此只会在一个节点上有正确的loss输出，使用上述PP并行可在启动文件`run_train.sh`将输出的rank号修改为如下形式，会在主节点上输出正常loss

```toml
export LOG_RANK=${LOG_RANK:-4}
```


在启动文件`run_train.sh`中修改config_file

```bash
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}
```

### 确定性计算配置说明

精度验证/对齐，需固定随机种子，设置确定性计算/通信

设置环境变量

```bash
#Set HCCL Determinism
export HCCL_DETERMINISTIC=true
#Set NPU Hardware Determinism
export ASCEND_LAUNCH_BLOCKING=1
```

在`torchtitan/train.py`中导包完成后，添加如下代码

```python
import torch_npu
import numpy as np
import random
seed = 1
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch_npu.npu.manual_seed_all(seed)
torch_npu.npu.manual_seed(seed)
```

### 环境变量与训练启动

激活环境变量，如果已激活，无需重复执行。

```bash
# source env of CANN, the real path on your server maybe different
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# If encounter a version mismatch issue between ATB and torch_npu, you can add and try switching the parameter --cxx_abi=0 or --cxx_abi=1.
source /usr/local/Ascend/nnal/atb/set_env.sh

# activate python env
```

运行脚本开始训练。

```bash
# cd torchtitan source root
bash ./run_train.sh
```

通过修改`run_train.sh`中`torchrun`命令可运行双机任务，训练时先运行主节点脚本再运行副节点脚本，具体修改方法如下

主节点配置
```bash
torchrun --nnodes 2 --node_rank 0 --local-ranks-filter ${LOG_RANK} --nproc_per_node=${NGPU} --master_addr="master_node_address" --master_port 29500 -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
```

副节点配置
```bash
torchrun --nnodes 2 --node_rank 1 --local-ranks-filter ${LOG_RANK} --nproc_per_node=${NGPU} --master_addr="master_node_address" --master_port 29500 -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
```