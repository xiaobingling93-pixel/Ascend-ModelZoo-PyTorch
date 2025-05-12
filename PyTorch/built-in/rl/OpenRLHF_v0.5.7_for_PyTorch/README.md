# OpenRLHF v0.5.7

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

OpenRLHF是一个基于Ray、DeepSpeed和HF Transformers构建的高性能RLHF框架。该代码实现OpenRLHF支持 `SFT` 和 `DPO` 训练，目前已验证支持模型为：`Qwen2-VL-2B-Instruct`。


# 准备训练环境

## 准备环境

- 推荐参考[配套资源文档](https://www.hiascend.com/developer/download/commercial)使用最新的配套版本。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 25.0.RC1.B115 </td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.0.RC1.B115 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.2.RC1.B010 </td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.6.0 </td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> 2.6.0 </td>
    </tr>
  </table>
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  1. 在模型源码包根目录下执行命令，安装模型对应的PyTorch版本需要的依赖。
     ```shell
      TARGET_DEVICE=NPU pip install -e .
     ```


  2. 在模型源码包根目录下执行命令,源码编译安装 transformers v4.51.0。
     ```shell
      git clone -b v4.51.0 https://github.com/huggingface/transformers.git
      cp transformers_need/modeling_qwen2_vl.py transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
      cd transformers
      pip install .
     ```
  
## 获取预训练模型

1. 用户自行下载 `Qwen2-VL-2B-Instruct`模型，通过参数 `--pretrain_path` 指定模型地址。

## 准备数据集

1. 模型源码包根目录下创建data文件夹，用户需自行下载 `llava-en-zh-300k` 和 `RLHF-V` 数据集，结构如下：

   ```
   data/
   ├── llava-en-zh-300k
   ├── RLHF-V
   ```

# 开始训练

1. 进入解压后的源码包根目录。

    ```shell
    cd /${模型文件夹名称}
    ```

2. 运行训练脚本。

    - 8卡SFT训练
      
        启动8卡训练
      
        ```shell
        bash test/train_qwen2_vl_sft_full_8p.sh --pretrain_path=xxxx --dataset_path=data/llava-en-zh-300k/zh  # 8p精度

        bash test/train_qwen2_vl_sft_performance_8p.sh --pretrain_path=xxxx --dataset_path=data/llava-en-zh-300k/zh  # 8p性能
        ```

    - 8卡DPO训练
      
        启动8卡训练
      
        ```shell
        bash test/train_qwen2_vl_dpo_full_8p.sh --pretrain_path=xxxx --dataset_path=data/RLHF-V  # 8p精度

        bash test/train_qwen2_vl_dpo_performance_8p.sh --pretrain_path=xxxx --dataset_path=data/RLHF-V  # 8p性能
        ```

# 训练结果展示

**表 2**  训练结果展示表（性能）

| MODEL | NAME                    | METHOD | Second Per Step(s) |
|:------------------------|:------------------------|:----------:|:----------------------:|
| Qwen2-VL-2B-Instruct                  | 8P-竞品A                  |   SFT   |           0.17108           |
| Qwen2-VL-2B-Instruct                  | 8P Atlas 200T A2 Box16  |   SFT   |           0.24064           |
| Qwen2-VL-2B-Instruct                  | 8P-竞品A                  |  DPO   |           0.40857           |
| Qwen2-VL-2B-Instruct                  | 8P Atlas 200T A2 Box16 |  DPO   |           0.53905           |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md。

# 版本说明

## 变更

2025.5.12：首次发布。

## FAQ