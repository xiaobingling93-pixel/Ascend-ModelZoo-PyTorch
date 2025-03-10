# 拉起多卡训练脚本示例

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [版本说明](#版本说明)



# 概述

## 简述

基于示例模型脚本，针对shell脚本启动、torch.distributed.launch启动、mp.spawn启动、torchrun启动、torch_npu_run启动五种方式，分别提供单机八卡和双机十六卡端到端训练示例脚本。


# 准备训练环境

## 准备环境
  
- 环境准备指导。

  请参考《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt  # PyTorch2.1版本
  ```


## 准备数据集

首次运行任意脚本时会自动下载MNIST数据集，运行成功后会在本地缓存数据，后续运行时会自动加载本地缓存数据，无需自行准备。


# 开始训练

## 训练模型

1. 进入项目文件目录

   ```
   cd ModelZoo-PyTorch/PyTorch/built-in/docs/dist_train_demo/
   ```

2. 运行训练脚本。


   - shell脚本启动。

     启动单机八卡训练。

     ```
     bash start_train_8p.sh
     ```
     启动双机十六卡训练。

     ```
     bash start_train_16p_master_addr.sh  # 主节点
     
     bash start_train_16p_slave_addr.sh  # 从节点
     ```
   - torch.distributed.launch脚本启动。

     启动单机八卡训练。

     ```
     python -m torch.distributed.launch --nproc_per_node 8 --master_addr localhost --master_port 12345 train_8p_torch_distributed_launch.py
     ```
     启动双机十六卡训练（"xxxx"替换为主节点IP地址）。

     ```
     python -m torch.distributed.launch --nnodes 2 --node_rank 0 --nproc_per_node 8 --master_addr xxxx --master_port 12345 train_16p_torch_distributed_launch.py  # 主节点
     
     python -m torch.distributed.launch --nnodes 2 --node_rank 1 --nproc_per_node 8 --master_addr xxxx --master_port 12345 train_16p_torch_distributed_launch.py  # 从节点
     ```
   - mp.spawn脚本启动（"xxxx"替换为主节点IP地址）。

     启动单机八卡训练。

     ```
     export MASTER_ADDR=XXXX
     export MASTER_PORT=12345
     python train_8p_spawn.py
     ```
     启动双机十六卡训练。

     ```
     # 主节点命令
     export MASTER_ADDR=XXXX
     export MASTER_PORT=12345
     python train_16p_spawn.py --nnodes 2 --node_rank 0

     # 从节点命令
     export MASTER_ADDR=XXXX
     export MASTER_PORT=12345
     python train_16p_spawn.py --nnodes 2 --node_rank 1
     ```

   - torchrun脚本启动。

     启动单机八卡训练。

     ```
     torchrun --nproc_per_node=8 --master_addr localhost --master_port 12345 train_8p_torchrun.py
     ```
     启动双机十六卡训练（"xxxx"替换为主节点IP地址）。

     ```
     torchrun --nnodes 2 --node_rank 0 --nproc_per_node 8 --master_addr xxxx --master_port 12345 train_16p_torchrun.py  # 主节点
     
     torchrun --nnodes 2 --node_rank 1 --nproc_per_node 8 --master_addr xxxx --master_port 12345 train_16p_torchrun.py  # 从节点
     ```
   - torch_npu_run脚本启动（"xxxx"替换为主节点IP地址）。

     启动单机八卡训练。

     ```
     torch_npu_run --rdzv_backend parallel --master_addr xxxx --master_port 12345 --nproc_per_node=8 train_8p_torch_npu_run.py
     ```
     启动双机十六卡训练。

     ```
     torch_npu_run --rdzv_backend parallel --master_addr xxxx --master_port 12345 --nnodes 2 --node_rank 0 --nproc_per_node=8 train_16p_torch_npu_run.py  # 主节点
     
     torch_npu_run --rdzv_backend parallel --master_addr xxxx --master_port 12345 --nnodes 2 --node_rank 1 --nproc_per_node 8 train_16p_torch_npu_run.py  # 从节点
     ```

# 版本说明

## 变更

2025.02.27：首次发布

## FAQ

无。
