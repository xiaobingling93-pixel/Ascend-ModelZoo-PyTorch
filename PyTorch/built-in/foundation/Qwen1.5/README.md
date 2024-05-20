# Qwen1.5 for PyTorch

## 模型描述

通义千问是阿里云研发的通义千问大模型系列。Qwen1.5是Qwen2的beta版本, 基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

## 支持任务列表
本仓已经支持以下模型任务类型

|      模型      | 任务列表 | 是否支持 |
|:------------:|:----:| :------: |
| Qwen1.5 0.5B|  微调  |    ✔     |
| Qwen1.5 1.8B|  微调  |    ✔     |
| Qwen1.5 4B|  微调  |    ✔     |
| Qwen1.5 7B|  微调  |    ✔     |
| Qwen1.5 14B|  微调  |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/QwenLM/Qwen1.5
  commit_id=da2c2d34b3a4f7d44a98c5421198c749dd912b96
  ```
- 适配昇腾AI处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/foundation
    ```

## 仓库介绍

`Qwen1.5` 基于 `PyTorch` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   Qwen1.5
     ├── script
        ├── finetune.py
   ```

2. 模型配置：

   ```text
   Qwen1.5
      ├── script
          ├── finetune0.5B         # 0.5B 全参微调启动配置
          ├── finetune1.8B         # 1.8B 全参微调启动配置
          ├── finetune4B           # 4B 全参微调启动配置
          ├── finetune7B           # 7B 全参微调启动配置
          └── finetune14B          # 14B 全参微调启动配置
   ```

3. 环境准备和任务启动脚本：

   ```text
   Qwen1.5
     ├── alpaca_converter.py           # alpaca数据集格式转换脚本
     ├── ds_config_zero2.json          # 配置文件
     ├── requirements                  # 环境准备
     ├── finetune.py                   # 源码文件
     └── script                        # 启动脚本
   ```

## 前期准备

### 环境要求

Python：3.9

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。

  **表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.RC1  |
|       CANN        | 8.0.RC1  |
|    昇腾NPU固件    | 24.1.RC1 |
|    昇腾NPU驱动    | 24.1.RC1 |

  **表 2**  三方库版本支持表

|    三方库    |  支持版本  |
|:---------:|:------:|
|  PyTorch  | 2.1.0  |
| accelerate |  0.29.0   |
| deepspeed |  0.14.0   |
| transformers |  4.39.2   |

在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖
```shell
pip install -r requirements.txt
```

### 模型权重准备

- Qwen1.5-0.5B-Chat
- Qwen1.5-1.8B-Chat
- Qwen1.5-4B-Chat
- Qwen1.5-7B-Chat
- Qwen1.5-14B-Chat

在当前目录下创建文件夹Qwen,下载权重文件后文件夹结构如下所示：

```text
Qwen1.5
  ├── alpaca_converter.py
  ├── ds_config_zero2.json
  ├── Qwen
      ├── Qwen1.5-0.5B-Chat
      ├── Qwen1.5-1.8B-Chat
      ├── Qwen1.5-4B-Chat
      ├── Qwen1.5-7B-Chat
      ├── Qwen1.5-14B-Chat
...
```


### 数据集准备

当前使用alpaca数据集的预处理脚本用于全参微调任务。

下载alpaca_data数据集并放置到源码包根目录下：

执行`alpaca_converter.py`，将原始数据集转换为指定格式并保存在当前目录下。

``` bash
python alpaca_converter.py 
```
新生成的文件名为alpaca_converter.json。
目录结构如下所示：
```text
Qwen1.5
  ├── alpaca_converter.json
```
转换后格式样例：

```text
  {
    "type": "chatml",
    "messages": [
      {
        "role": "user",
        "content": "Give three tips for staying healthy."
      },
      {
        "role": "assistant",
        "content": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ],
    "source": "alpaca"
  },
```

## 执行微调
1) 执行以下脚本，进入目录script
```
  cd script
```
2) 执行训练，如下命令：
```
  bash finetune0.5B.sh     # 保存日志可执行bash finetune0.5B.sh > Qwen0.5.txt重定向日志到当前目录Qwen0.5.txt文件下
```
3) 执行14B训练时，需要使用集群双机：
需要将config.yaml文件中main_process_ip改为主机的ip，并且将副机中的machine_rank设置为1
在两台机器上先后执行以下脚本
```
  bash finetune14B.sh
```

## 训练结果
训练loss与train_samples_per_second可在训练日志中获取，其结果如下：

  **表 3**  训练结果展示表

| 芯片       | 模型       | 卡数       | Batch size | Steps | Train_Samples_Per_Second |
|----------|:--------:|:--------:|:----------:|:-----:|:------------------------:|
| GPU      | 0.5B       |    8p    |     2      | 2000  |          62.092          |
| GPU      | 1.8B       |    8p    |     2      | 2000  |          45.856          |
| GPU      | 4B       |    8p    |     2      | 2000  |          29.448          |
| GPU      | 7B       |    8p    |     2      | 2000  |          20.367          |
| GPU      | 14B       |    16p    |     2      | 2000  |          14.925          |
| Atlas A2    | 0.5B       |    8p    |     2      | 2000  |          60.367          |
| Atlas A2      | 1.8B       |    8p    |     2      | 2000  |          56.159          |
| Atlas A2      | 4B       |    8p    |     2      | 2000  |          31.485          |
| Atlas A2      | 7B       |    8p    |     2      | 2000  |          21.926          |
| Atlas A2      | 14B       |    16p    |     2      | 2000  |          22.356          |

# 公网地址变更说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明
2024.05.17：Qwen1.5 bf16微调任务首次发布。


# FAQ

暂无。

