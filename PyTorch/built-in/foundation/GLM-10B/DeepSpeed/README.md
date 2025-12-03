# 本仓仅用于需求/问题跟踪、测试用例。

## DeepSpeed 现已原生支持 NPU，本仓已不做代码维护 ！

deepspeed==0.10.0 及之后版本无需 deepspeed_npu 插件，直接安装，直接使用，建议使用新版 DeepSpeed。

1. 首先卸载旧版 DeepSpeed 以及 deepspeed_npu（如没有可跳过此步骤）。
    ```shell
    pip3 uninstall deepspeed_npu
    pip3 uninstall deepspeed
    ```

2. 安装最新版本 DeepSpeed：
   
    方式一：
    ```shell
    pip3 install deepspeed
    ```

    方式二：
    ```shell
    git clone https://github.com/microsoft/DeepSpeed.git
    cd DeepSpeed
    pip3 install -e ./
    ```

3. 使用方式与原生一致，具体可参考官方文档与示例。  
    官方文档：http://www.deepspeed.ai/  
    官方代码仓：https://github.com/microsoft/DeepSpeed   
    官方示例仓：https://github.com/microsoft/DeepSpeedExamples


#### 以下为原文:
# deepspeed_npu

Ascend NPU 适配 Deepspeed 插件

# 简介

通过 deepspeed_npu，你可以在 Ascend910 芯片上使用 Deepspeed，并基于 Deepspeed 进行开发。

# 安装

deepspeed_npu 目前仅支持 Deepspeed 版本 0.9.2：https://github.com/microsoft/DeepSpeed/tree/v0.9.2
 
#### 1. 先安装原生 Deepspeed

```bash
pip3 install deepspeed==0.9.2
```

#### 2. 然后安装 deepspeed_npu 插件

```bash
git clone https://gitcode.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install .
```

#### 3. 卸载方法

作为 Python 包，deepspeed_npu 与其他 python 包一样，可通过 pip 命令卸载：

```shell
pip uninstall deepspeed_npu
```

# 快速上手

在模型启动文件中 import deepspeed_npu，并配合 deepspeed / torch 使用,例如

```python

import torch
import torch_npu
import deepspeed
import deepspeed_npu
...
```

# 特性介绍

目前，deepspeed_npu 主要支持以下特性:

1. FP16
2. Gradient Accumulation
3. Data Parallelism
4. Pipeline Parallelism
5. Tensor Parallelism (Inference Engine)
6. ZeRO (stage1-stage3)
7. Activation Checkpointing
8. ZeRO-Offload
9. CPU Adam
10. Fused Adam
11. One-bit Adam
12. MoE
13. Zero Infinity
14. Zero-One Adam
15. Curriculum Learning
16. Progressive layer dropping

请参考 Deepspeed 官方文档获取这些特性的详细说明：[https://www.deepspeed.ai/](https://www.deepspeed.ai/)

DeepSpeed 用例参考: [https://github.com/microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)


# 关于

### 目录结构

- `deepspeed_npu`：文件夹下的各个文件都对应原生的文件，如 adaptor_xxx_yyy.py 文件对应原生的 xxx.yyy.py 文件。
- `deepspeed_npu.csrc_npu`：文件夹下为相关特性的动态编译 C++ 文件，与原生的 csrc 文件夹相对应。如 csrc_npu/adam 文件夹对应原生的 csrc/adam 文件夹。

### 接口替换

deepspeed_npu 以 monkey patching / 装饰器等方式替换/修改 DeepSpeed 原有函数实现，并不提供对外接口，用户只需要`import deepspeed_npu`，做到无感迁移原有模型代码。


# 安全声明

[deepspeed_npu 安全声明](SECURITYNOTE.md)
