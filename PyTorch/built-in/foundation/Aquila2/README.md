# Aquila2 for Pytorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [代码实现](#代码实现)
-   [Aquila2](#Aquila2)   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [预训练任务](#预训练任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

Aquila2是智源发布的业内领先的大语言模型，在多个领域都有着广泛的应用前景，如自然语言处理、机器翻译、智能问答等。Aquila2在多个公开数据集上的表现都非常优秀，是当前自然语言处理领域的前沿技术之一。

## 代码实现

- 参考实现：

  ```
  url=https://github.com/FlagOpen/FlagScale.git
  commit_id=d7dc60ec3ef6341526fd187281dc289418c17899
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```


# Aquila2

## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  | 三方库    | 支持版本  |
  | :--------: | :-------------: |
  | PyTorch | 2.1.0 |
  | transformers | 4.32.0 |
  | torchvision | 0.16.0 | 


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```python
   pip install -r ascend/requirements.txt
   ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

  | 软件类型   | 支持版本  |
  | :--------: | :-------------: |
  | FrameworkPTAdapter | 在研版本 |
  | CANN | 在研版本 |
  | 昇腾NPU固件 | 在研版本 | 
  | 昇腾NPU驱动 | 在研版本 |

  

### 准备数据集

#### 预训练数据集准备

1. 用户自行获取原始数据集，以Pile数据集为例，数据集目录结构如下：
   ```
   pile_wikipedia_demo
   ├── pile_wikipedia_demo.idx
   └── pile_wikipedia_demo.bin
   ```

2. 获取对应数据集后，在以下启动shell脚本中将`data_path`参数设置为本地数据集的绝对路径。

   ```shell
   ascend/scripts/pretrain_aquila_34B_distributed.json
   ascend/scripts/pretrain_aquila_70B_distributed.json
   ```


## 快速开始

### 预训练任务

本任务主要提供**bf16**的训练脚本，默认使用**megatron**分布式训练。

#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 根据实际运行环境修改以下shell脚本中的对应参数。
   * `shell_cmds`：环境变量依赖；
   * `ssh port`：ssh服务端口；
   * `GLOO_SOCKET_IFNAME`: 设置为服务器网卡名；
   * `HCCL_SOCKET_IFNAME`: 设置为服务器网卡名；

   ```
   ascend/scripts/pretrain_aquila_34B_distributed.json
   ascend/scripts/pretrain_aquila_70B_distributed.json
   ```

3. 运行预训练脚本。

   ```
   # 34b 16卡训练
   bash ./test/pretrain_aquila2_34b.sh --extra-config=./ascend/scripts/pretrain_aquila_34B_distributed_extra_16p.json --hostfile='./hostfile'
   # 34b 16卡训练性能
   bash ./test/pretrain_aquila2_34b.sh --extra-config=./ascend/scripts/pretrain_aquila_34B_distributed_extra_16p.json --hostfile='./hostfile' --mode=performance
   
   # 70b 32卡训练
   bash ./test/pretrain_aquila2_70b.sh --extra-config=./ascend/scripts/pretrain_aquila_70B_distributed_extra_32p.json --hostfile='./hostfile'
   # 70b 32卡训练性能
   bash ./test/pretrain_aquila2_70b.sh --extra-config=./ascend/scripts/pretrain_aquila_70B_distributed_extra_32p.json --hostfile='./hostfile' --mode=performance
   ```
   * 多机训练场景下需要传入参数`hostfile`，该文件中列举了多机场景涉及的服务器IP，每行一个。
   * 模型训练日志默认保存在`test/aquila_{参数规模}_{卡数}`路径下。

4. 获取模型性能。
   ```
   bash ./test/parse_throughout.sh --log=xxx
   ```
   * `log`需要传入模型训练的日志文件路径（若涉及多机场景，则传入`hostfile`中最后一台机器的日志文件路径）
   
#### 训练结果


 **表 3**  训练结果展示表
| 芯片 | 卡数 | 参数规模 | seq_length | micro_batch_size | global_batch_size | 单步迭代时间 (s/step) | tokens吞吐 (tokens/s/p)
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPU | 16p | 34B | 4096 | 1 | 32 | 10.8 | 756 |
| Atlas A2 | 16p | 34B | 4096 | 2 | 64 | - | - |
| GPU | 32p | 70B | 4096 | 1 | 44 | - | - |
| Atlas A2 | 32p | 70B | 4096 | 1 | 44 | - | - |

# 变更说明

## 变更

2024.04.15：首次发布。

# FAQ

暂无。

