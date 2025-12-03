# OpenSora1.0 for PyTorch
**注意**： 本仓库OpenSora1.0模型将不再进行维护，请使用[MindSpeed-MM](https://gitcode.com/ascend/MindSpeed-MM/tree/master/examples/opensora1.0)

# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [STDiT（在研版本）](#STDiT（在研版本）)
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [训练任务](#训练任务)
          - [推理任务](#推理任务)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介
## 模型介绍

OpenSora是HPC AI Tech开发的开源高效复现类Sora视频生成方案。OpenSora不仅实现了先进视频生成技术的低成本普及，还提供了一个精简且用户友好的方案，简化了视频制作的复杂性。
本仓库主要将STDiT模型的任务迁移到了昇腾NPU上，并进行极致性能优化。

> <span style="color: red;">**注意**: OpenSora1.0目录下面的OpenSora1.0模型已经集成到[MindSpeed-MM](https://gitcode.com/ascend/MindSpeed-MM)中,当前目录下的OpenSora1.0模型不再维护，MindSpeed-MM是面向大规模分布式训练的昇腾多模态大模型套件，同时支持多模态生成及多模态理解,欢迎大家使用。</span>

## 支持任务列表

本仓已经支持以下模型任务类型

|  模型  | 任务列表 | 是否支持 |
|:----:|:----:|:-----:|
| STDiT-XL/2 |  训练  | ✔ |
| STDiT-XL/2 | 在线推理 | ✔ |
| STDiT-XL/2 | 序列并行 | ✔ |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/hpcaitech/Open-Sora
  commit_id=436ee2c91faee50f925d80f5148b36a4f820d1e3
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mm/OpenSora1.0
  ```


# STDiT（在研版本）

## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  |     三方库     |  支持版本  |
  |:-----------:|:------:|
  |   PyTorch   | 2.1.0  |
  | TorchVision | 0.16.0 |


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```python
   source ${cann_install_path}/ascend-toolkit/set_env.sh              # 激活cann环境，默认在/usr/local/Ascend下
   cd OpenSora1.0
   pip install -v -e .                                                # 安装本地代码仓，同时自动安装依赖

   # 以https://gitee.com/aijgnem/MindSpeed最新文档为准，安装 MindSpeed
   git clone https://gitcode.com/ascend/MindSpeed.git
   cd MindSpeed
   git checkout 3e7d2377f1947594708ced2fe66f6428da9d330f
   cd ..
   pip install -e MindSpeed

   # 以https://gitee.com/aijgnem/MindSpeed最新文档为准，获取 Megatron-LM 并指定 commit id
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   # 注意：启动脚本PYTHONPATH添加Megatron-LM路径，如 export PYTHONPATH=$PYTHONPATH:./Megatron-LM
   git checkout core_r0.6.0
   cd ..
   ```
### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。


  **表 2**  昇腾软件版本支持表

| 软件类型   |   支持版本   |
| :--------: |:--------:|
| FrameworkPTAdapter |   在研版本   |
| CANN |   在研版本   |
| 昇腾NPU固件 |   在研版本   |
| 昇腾NPU驱动 | 在研版本 |



### 准备数据集

#### 训练数据集准备

1. 用户需自行获取并解压MSRVTT数据集。

2. 在源码根目录下进行数据集预处理。

    ```shell
    python tools/datasets/collate_msr_vtt_dataset.py -d ${MSRVTT原数据集的路径} -o ${MSRVTT处理后数据集的路径}
    python tools/datasets/preprocess_msrvtt.py --data_path ${MSRVTT处理后数据集的路径}/train/annotations.json   # 生成最终的标注csv文件
    ```

3. 在以下配置文件中将`root`参数设置为本地数据集的绝对路径, 将`data_path`参数设置为数据集的标注csv文件的绝对路径，如`${MSRVTT处理后数据集的路径}/train/annotations.csv`。

   ```shell
   configs/opensora/train/16x256x256.py
   configs/opensora/train/120x256x256.py
   ```

   数据结构如下：

   ```
   $MSRVTT
   ├── train
   ├── ├── videos
   ├── ├── ├── video0.mp4
   ├── ├── ├── video1.mp4
   ├── ├── ├── ...
   ├── ├── annotation.csv
   ├── val
   ├── test
   └── ...
   ```

   > **说明：**
   > 该数据集的训练过程脚本只作为一种参考示例。


### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   PixArt-alpha/PixArt-alpha   # PixArt-XL-2-512x512模型(训练用)
   stabilityai/sd-vae-ft-ema   # vae模型
   DeepFloyd/t5-v1_1-xxl       # t5模型
   hpcai-tech/Open-Sora        # 预训练权重(推理用)
   ```

3. 获取对应的预训练模型后，在以下配置文件中将`model`、`vae`的`from_pretrained`参数设置为本地预训练模型绝对路径。
   ```shell
   configs/OpenSora/train/16x256x256.py
   configs/OpenSora/train/120x256x256.py
   configs/OpenSora/inference/120x256x256.py
   ```

4. 将下载好的t5模型放在本工程目录下的`DeepFloyd`目录下，组织结构如下：
   ```
   $OpenSora1.0
   ├── DeepFloyd
   ├── ├── t5-v1_1-xxl
   ├── ├── ├── config.json
   ├── ├── ├── pytorch_model-00001-of-00002.bin
   ├── ├── ├── ...
   └── ...
   ```

## 快速开始

### 训练任务

本任务主要提供**混精bf16**的**8卡**训练脚本。

#### 开始训练
1. 进入解压后的源码包根目录。

   ```shell
   cd /${模型文件夹名称}
   ```

2. 运行预训练脚本。

   该模型支持单机8卡训练。


   - 单机8卡训练

     ```shell
     bash test/train_full_8p_bf16.sh # 8卡训练，混精bf16
     bash test/train_full_8p_bf16.sh --max_train_steps=200 # 8卡性能，混精bf16
     ```

     模型训练python训练脚本参数说明如下。

     ```
     scripts/train.py
     config                               //配置文件路径
     --seed                               //随机种子
     --data_path                          //数据集标注csv文件路径
     --batch_size                         //设置batch_size
     --max_train_steps                    //最大训练步数，默认是0，不会提前停止。
     ```

   - 序列并行(以120x256x256的训练任务为示例)

     若要使能序列并行，需要修改配置文件：configs/opensora/train/120x256x256-sp.py

     - 添加enable_sequence_parallelism

     ```python
     # 修改前
     # Define model
     model = dict(
        type="STDiT-XL/2",
        space_scale=0.5,
        time_scale=1.0,
        from_pretrained="PixArt-XL-2-512x512.pth",
        enable_flashattn=True,
        enable_layernorm_kernel=True,
     )

     # 修改后，增加enable_sequence_parallelism=True：
     model = dict(
         type="STDiT-XL/2",
         space_scale=0.5,
         time_scale=1.0,
         from_pretrained="PixArt-XL-2-512x512.pth",
         enable_flashattn=True,
         enable_layernorm_kernel=True,
         enable_sequence_parallelism=True,
     )
     ```

     - 增加序列并行其他配置

     ```python
     sp_size = 8
     context_parallel_algo = 'megatron_cp_algo'
     use_cp_send_recv_overlap = True
     ```

     参数说明：

     ```
     sp_size: 序列并行大小，当sp_size设置为1时，将不会使能序列并行
     use_cp_send_recv_overlap：是否开启序列并行send recv overlap, 仅在context_parallel_algo设置为'megatron_cp_algo'有效
     context_parallel_algo:设置为'megatron_cp_algo'表示序列并行使用ring attention算法, 设置为"ulysses_cp_algo"表示序列并行算法使用ulysses算法, 设置为"dsp_cp_algo"表示序列并行算法使用dsp算法
     ```

     即可，之后按照前面提及的单机八卡训练任务开展训练。

   - VAE序列并行(以120x256x256的训练任务为示例)

      若要使能VAE序列并行，需要修改配置文件：configs/opensora/train/120x256x256-dsp.py

     - 添加enable_sequence_parallelism

      ```python
      # 修改前
      vae = dict(
         type="VideoAutoencoderKL",
         from_pretrained="stabilityai/sd-vae-ft-ema",
      )

      # 修改后，增加enable_sequence_parallelism=True：
      vae = dict(
         type="VideoAutoencoderKL",
         from_pretrained="stabilityai/sd-vae-ft-ema",
         enable_sequence_parallelism=True,
      )
      ```

- 梯度累积：在配置文件中指定global_batch_size变量即可



#### 训练结果


##### 性能
| 芯片 | 卡数 |  FPS  | batch_size | sp_size | AMP_Type | Torch_Version |
|:---:|:---:|:-----:|:----------:|:-------:|:---:|:---:|
| 竞品A | 8p | 3.56  |     8      |    1    | bf16 | 2.1 |
| Atlas 800T A2 | 8p | 2.35  |     8      |    1    | bf16 | 2.1 |
| 竞品A-sp | 8p | 0.885 |     1      |    8    | bf16 | 2.1 |
| Atlas 800T A2-dsp + vae sp | 8p | 0.68  |     1      |    8    | bf16 | 2.1 |
| 竞品A-sp | 8p | 2.039 |     2      |    4    | bf16 | 2.1 |
| Atlas 800T A2-dsp + vae sp | 8p | 1.951 |     2      |    4    | bf16 | 2.1 |



### 推理任务
本任务主要以预训练模型为主，展示推理任务，包括单卡在线推理。
#### 开始推理
1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称}
   ```


2. 运行推理的脚本。

- 单机单卡推理
  ```shell
  bash test/infer_full_1p.sh # 混精fp16 在线推理
  ```
- 微调脚本参数说明如下
   ```shell
   scripts/inference.py
   config                               //配置文件路径
   --seed                               //随机种子
   --data_path                          //数据集标注csv文件路径
   --batch_size                         //设置batch_size
   --prompt_path                        //推理使用的prompt文件路径
   --save_dir                           //输出视频的路径
   --num_sampling_steps                 //推理的采样步数
   --cfg_scale                          //无分类器引导的权重系数
   ```



# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.05.11：OpenSora 1.0 bf16训练和fp16推理任务首次发布。


# FAQ



