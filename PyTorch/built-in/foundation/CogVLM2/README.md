# CogVLM2 for Pytorch
# 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [CogVLM2](#CogVLM2)
  - [准备训练环境](#准备训练环境)
  - [准备数据集](#准备数据集)
  - [快速开始](#快速开始)
  - [微调任务](#微调任务)
  - [在线推理任务](#在线推理任务)
- [公网地址变更说明](#公网地址变更说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)



# 简介
## 模型介绍

官方仓：https://github.com/THUDM/CogVLM2

说明：本仓代码仅为适配官方仓脚本，执行训练与推理需要在官方仓cogvlm2项目路径下进行。

## 支持任务列表
本仓已支持以下模型任务类型。

| 模型      | 模型大小 | 任务类型   | 是否支持  |
|---------|------|--------| ------------ |
| CogVLM2 |   cogvlm2-llama3-chinese-chat-19B   | lora微调 | ✅   |

## 代码实现
- 参考实现
  ```
  CogVLM仓: https://github.com/THUDM/CogVLM
  commit id: 3adb5ce3243a9c81c1df5336d3297c94d0f9e1cc
  参考链接：https://github.com/THUDM/CogVLM2/tree/3adb5ce3243a9c81c1df5336d3297c94d0f9e1cc
  ```
- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation/CogVLM2
  ```
  
# CogVLM2

## 准备训练环境
### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|        软件类型        |   支持版本   |
|:------------------:|:--------:|
| FrameworkPTAdapter | 6.0.RC3  |
|        CANN        | 8.0.RC3  |
|      昇腾NPU固件       | 24.1.RC3 |
|      昇腾NPU驱动       | 24.1.RC3 |

### 安装模型环境

**表 2**  三方库版本支持表

|    三方库    |  支持版本  |
|:---------:|:------:|
|  PyTorch  | 2.1.0  |

1) 安装模型对应PyTorch版本需要的依赖, 需要先安装PTA包。
2) 下载model_zoo下面的CogVLM2相关文件，可以根据cogvlm2官方仓安装三方件或者使用本仓提供的requirements.txt进行三方件安装。
```shell
pip install -r requirements.txt
```
### 准备数据集

1) 微调数据集:
训练与评估所使用的数据集为CogVLM-SFT-311K[下载](https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K)，该数据集是官网提供的一个lora微调数据集。
2) 数据集准备：
进行微调时，可以根据官方提醒指定到具体文件，如CogVLM-SFT-311K/llava_instruction_multi_conversations_formate/

### 获取预训练权重

1) 官方提供微调权重cogvlm2-llama3-chinese-chat-19B[下载](https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B/)。


## 快速开始

### 微调任务
主要提供基于CogVLM-SFT-311K数据集微调的lora微调训练脚本。
#### 模型适配

替换和新增本仓authority_repository/finetune_demo下的文件到cogvlm2官方仓的finetune_demo/文件下
替换和新增本仓llama3_chinese_chat_19B下的文件到预训练权重文件下

#### 执行单机8卡微调
1) 替换authority_repository/finetune_demo/cogvlm2_lora_finetune.sh文件中的"训练数据路径"，"预训练权重路径"和"模型保存路径"为实际路径

2) CogVLM2官方仓下执行训练，如下命令：
```
cd finetune_demo
bash cogvlm2_lora_finetune.sh
```

##### 性能

|    芯片    | 卡数 | s/it | micro_batch_size | AMP_Type | Torch_Version |
|:--------:| :----: |:----:|:----------------:|:--------:| :-----------: |
|   GPU    |   8p   | 3.3  |        1         |   bf16   |      2.1      |
| Atlas A2 |   8p   |  4   |        1         |   bf16   |      2.1      |

#### 执行双机16卡微调
1) 替换authority_repository/finetune_demo/cogvlm2_lora_finetune_2nodes.sh文件中的"训练数据路径"，"预训练权重路径"和"模型保存路径"为实际路径
2) authority_repository/finetune_demo/hostfile中的内容为服务器名(server1和server2)和每台服务器的卡数
3) 双机配置可参考"配置双机通信环境"

2) CogVLM2官方仓下执行训练，如下命令：
```
cd finetune_demo
bash cogvlm2_lora_finetune_2nodes.sh
```

### 配置双机通信环境
1) 安装pdsh
url： https://github.com/chaos/pdsh/tree/pdsh-2.29
**安装**
```python
git clone https://github.com/chaos/pdsh/archive/refs/tags/pdsh-2.29.tar.gz

tar -zxvf pdsh-2.29.tar.gz
cd pdsh-2.29
./configure --with-ssh --with-rsh --with-mrsh --with-mqshel --with-qshell  --with-dshgroups --with-machines=/etc/pdsh/machines  --without-pam

make
make install
```

安装完成后，执行`pdsh -h`命令。显示如下信息，表示安装成功。
```shell
# pdsh -h
Usage: pdsh [-options] command ...
-S                return largest of remote command return values
-h                output usage menu and quit
-V                output version information and quit
-q                list the option settings and quit
-b                disable ^C status feature (batch mode)
-d                enable extra debug information from ^C status
-l user           execute remote commands as user
-t seconds        set connect timeout (default is 10 sec)
-u seconds        set command timeout (no default)
-f n              use fanout of n nodes
-w host,host,...  set target node list on command line
-x host,host,...  set node exclusion list on command line
-R name           set rcmd module to name
-M name,...       select one or more misc modules to initialize first
-N                disable hostname: labels on output lines
-L                list info on all loaded modules and exit
-g groupname      target hosts in dsh group "groupname"
-X groupname      exclude hosts in dsh group "groupname"
-a                target all nodes
available rcmd modules: ssh,rsh,exec (default: rsh)

```
2) 双机通信配置
首先，我们需要编辑两台服务器的/etc/hosts文件，添加两台服务器的IP地址，并将node1和node2替换为两台服务器的实际IP地址
``shell
vim /etc/hosts
```
```shell
node1 server1
node2 server2
```

然后，我们需要执行以下命令来生成sshkey。

```shell
ssh-keygen -t rsa
```
接着，将ssh-key拷贝到每个节点，本机也要拷贝。

```shell
ssh-copy-id root@server1
ssh-copy-id root@server2
```
然后，在每个节点上运行以下代码，首次执行时需要手动输入`yes`，然后执行`exit`退出。再次执行以下命令时，如果不需要输入密码，则表示配置成功。

```shell
ssh server1
ssh server2
```

#### 随机性说明
模型中包含多种随机问题，会影响loss曲线和下游任务，用户可根据需要自行修改，部分确定性问题本代码不做更换：
1) Cogvlm2项目路径的finetune_demo/peft_lora.py 中DataLoader是开启了shuffle，根据需要进行关闭：
2) 模型本身有确定性问题，需要固定随机种子
3) triton中的FastRotaryEmbedding和RotaryEmbedding精度上也略有差异


### 在线推理任务

#### 推理前准备
1) Cogvlm2项目路径的finetune_demo/peft_infer.py文件中在import torch后新增npu依赖，如下所示。
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
2) 替换peft_infer.py中MODEL_PATH和PEFT_MODEL_PATH为实际路径，执行推理
```shell
sh peft_infer.py
```

# 公网地址变更说明
暂无。

# 变更说明
2024.08.08：CogVLM2 bf16微调任务首次发布。

# FAQ

暂无。