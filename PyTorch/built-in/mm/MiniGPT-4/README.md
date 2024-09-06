# MiniGPT-4 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述
MiniGPT-4使用一个投影层将来自BLIP-2的冻结视觉编码器与冻结的LLM Vicuna对齐。通过两个阶段来训练MiniGPT-4，先是用500万图文对训练，然后再用一个3500对高质量数据集训练。

- 参考实现：
  ```
  url=https://github.com/Vision-CAIR/MiniGPT-4
  commit_id=22d8888ca2cf0aac862f537e7d22ef5830036808
  ```

- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的PyTorch如下表所示。

  **表 1**  版本支持表

  |  配套   |                           版本                           |
  | :-----: | :------------------------------------------------------: |
  | PyTorch | [1.11.0](https://gitee.com/ascend/pytorch/tree/v1.11.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt  # PyTorch1.11版本
  ```

- 替换transformers库中的相关文件。

  将当前工程目录下的transformers_modify文件夹中的文件替换到transformers安装目录下的对应位置（基于transformers 4.28.0版本）：
  ```
  utils.py -> transformers/generation/utils.py
  使用下面命令进行替换：
  cp ./transformers_modify/utils.py /path_to_transformers/transformers/generation/utils.py
  ```

## 准备数据集

1. 获取预训练数据集。

   要下载和准备Laion和CC数据集，请查看[第一阶段数据集准备说明](dataset/README_1_STAGE.md)。
   数据集参考目录如下:

   ```
   laion_dataset
   ├── 00000.parquet
   ├── 00000_stats.json
   ├── 00000.tar
   ├── ...

   cc_sbu_dataset
   ├── 00000.parquet
   ├── 00000_stats.json
   ├── 00000.tar
   ├── ...
   ```

2. 获取微调数据集

   要下载和准备小型高质量图像文本对数据集，请查看[第二阶段数据集准备说明](dataset/README_2_STAGE.md)。
   数据集参考目录如下:
   ```
   cc_sbu_align
   ├── filter_cap.json
   ├── image
      ├── 0.jpg
      ├── ...
   ```

## 准备模型权重和离线库

1. 准备预训练的Vicuna权重

   用户参照[链接](PrepareVicuna.md)自行获取模型文件，并放于自定义目录下，微调依赖该模型权重。
   自定义参考目录如下:
   ```
   vicuna_weights
   ├── config.json
   ├── generation_config.json
   ├── pytorch_model.bin.index.json
   ├── pytorch_model-00001-of-00003.bin
   ```
    在配置文件[minigpt4.yaml](minigpt4/configs/models/minigpt4.yaml#L16)中修改vicuna权重所在的路径。

2. 准备训练的MiniGPT-4检查点:

   |              Checkpoint Aligned with Vicuna 3B               |              Checkpoint Aligned with Vicuna 7B               |
   | :----------------------------------------------------------: | :----------------------------------------------------------: |
   | 官网下载 | 官网下载 |

   检查点数据请自行官网下载，并在评估配置文件[minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10)的第11行中设置预训练检查点的路径。

3. 准备只有第一阶段训练的MiniGPT-4检查点[链接](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link)。

4. 准备离线下载的bert-base-cased库，并在[blip2.py](minigpt4/models/blip2.py)的第31行和48行修改加载路径。

5. 准备离线下载的eva_vit_g.pth文件，并在[eva_vit.py](minigpt4/models/eva_vit.py)的第433行修改加载路径。

6. 准备离线下载的blip2_pretrained_flant5xxl.pth库，并在[mini_gpt4.py](minigpt4/models/mini_gpt4.py)的第27行和226行修改加载路径。


# 开始训练

  进入解压后的源码包根目录。
  ```bash
cd /${模型文件夹名称}
  ```

## 预训练

   - 单机4卡预训练
     ```bash
     bash test/pretrain_gpt_4p.sh
     ```
     要启动第一阶段预训练，请先在[laion/defaults.yaml](minigpt4/configs/datasets/laion/defaults.yaml)和[/cc_sbu/defaults.yaml](minigpt4/configs/datasets/cc_sbu/defaults.yaml)中指定预训练数据集路径。

## 微调
   - 单机单卡微调
     ```bash
     bash test/finetune_gpt_1p.sh
     ```
     要启动第二阶段微调对齐，请先在[minigpt4_stage2_finetune.yaml](train_configs/minigpt4_stage2_finetune.yaml)和[cc_sbu/align.yaml](minigpt4/configs/datasets/cc_sbu/align.yaml)中分别指定第1阶段预训练的检查点文件的路径和精调数据集路径。

## 在线演示

1. 修改配置文件[minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L11)第11行，路径为微调好的权重所在路径。

2. 在线演示：
   ```bash
   python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
   ```

3. 运行成功后，在服务器浏览器的输入URL链接：http://127.0.0.1:7860, 会加载UI界面。上传图像开始与MiniGPT-4聊天。

4. 如需本地浏览器远程访问服务器，需要ssh进行端口映射：
   ```bash
   ssh -L 6006:127.0.0.1:7860 yourname@server.ip
   ```
   在本地浏览器输入URL链接：http://127.0.0.1:6006, 即可加载聊天界面。

# 训练结果展示

**表 1**  预训练结果展示表

|     NAME      | TokensPerSec | Iterations | BatchSize | Torch_Version |
| :-----------: | :----------: | :--------: | :-------: | :-----------: |
|     竞品A     |     8866     |   5000*4   |    64     |     1.11      |
| Atlas 800T A2 |     7517     |   8000*4   |    40     |     1.11      |

**表 2**  微调结果展示表

|     NAME      | TokensPerSec | Iterations | BatchSize | Torch_Version |
| :-----------: | :----------: | :--------: | :-------: | :-----------: |
|     竞品A     |     2773     |   240*2    |    12     |     1.11      |
| Atlas 800T A2 |     2513     |   240*2    |    12     |     1.11      |

## 在线演示效果

这里展示了MiniGPT-4微调后的演示效果。

![demo](figs/ad_1.jpg)

![demo](figs/ad_2.jpg)

![demo](figs/fact_1.jpg)

![demo](figs/fun_1.jpg)


# 版本说明

## 变更

2023.7.05：首次发布。

2024.6.14：对ReadMe进行完善和补充，并修改代码中的一些问题。

## FAQ

### 端口问题

执行脚本中端口设置为12345，不同的机器执行时可能会出现端口占用或者不能运行问题，修改端口号即可，如将12345改为7900。如下所示：

```bash
#! /bin/bash
source test/env_npu.sh
export HCCL_CONNECT_TIMEOUT=6000
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=7900
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
OPTIONS="run.max_epoch=2 run.iters_per_epoch=240 run.batch_size_train=12 run.batch_size_eval=12 "
torchrun $DISTRIBUTED_ARGS train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml --options ${OPTIONS} | tee npu_fine_log.txt
```

执行代码强制结束后重新运行，会出现端口占用问题，需要kill相关进程，以及修改端口。同时修改后会出现命令不能识别问题，需要set ff=unix修改文件格式。此处提供脚本flush.sh可以自动更新端口和kill相关进程：

```bash
#! /bin/bash
file="test/finetune_gpt_1p.sh"
line=$(sed -n '9p' $file)
# 从文本中提取等号右边的数字
num=$(echo $line | cut -d'=' -f2)
# 让数字自增1
num=$((num + 1))
# 构建新的行文本并替换第9行的内容
new_line="MASTER_PORT=${num}"
sed -i "9s/.*/$new_line/" $file
echo "已将finetune_gpt_1p第9行的num值自增1"

# kill相关进程
pkill -9 python3.8
pkill -9 python
pkill -9 torchrun
pkill -9 bash
```

### 自动混合精度GradScaler

npu版本为控制loss的缩放比例，避免浮点数溢出，开启GradScaler的自动混合精度，将dynamic设为True，并设置缩放比例，让其动态对齐，在[runner_base.py](minigpt4/runner/runner_base.py)中的第137行进行修改，修改方式如下：

```python
self._scaler = torch.cuda.amp.GradScaler(dynamic=True, init_scale=2**16)
```

### numpy版本不适配问题

当出现numpy版本不适配导致模型卡住不能运行时，将numpy卸载（uninstall）并重新安装（install）即可。

```bash
pip uninstall numpy
pip install numpy
```