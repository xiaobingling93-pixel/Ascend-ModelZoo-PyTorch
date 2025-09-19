# Ling_mini_v2 模型 LLama Factory 训练环境搭建

- [硬件要求](#硬件要求)
- [前置操作](#前置操作)
- [环境搭建](#环境搭建)
- [数据集](#数据集)
- [模型权重](#模型权重)
- [训练配置](#训练配置)
- [启动训练](#启动训练)

## 硬件要求

Ling-V2的参考硬件配置如下,本文将以Atlas 800T A2单机8卡、A3单机8卡，SFT为例进行介绍：

| 类型 | 硬件 |        配置         |
| :--: | :--: | :-----------------: |
| 训练 | NPU  | 8 x Ascend NPUs(A2) |
| 训练 | NPU  | 8 x Ascend NPUs(A3) |

## 前置操作

主要依赖配套如下表，版本信息以如下表格信息为主。

| 依赖软件                                    | 版本     |
| :------------------------------------------ | :------- |
| 昇腾NPU驱动                                 | 商发版本 |
| 昇腾NPU固件                                 |          |
| CANN Toolkit（开发套件）                    | 8.2.RC1  |
| CANN Kernel（算子包）                       |          |
| CANN NNAL（Ascend Transformer Boost加速库） |          |
| Python                                      | >=3.10   |
| PyTorch                                     | 2.5.1    |
| torch_npu插件                               | 2.5.1    |
| transformers                                | 4.52.3   |

[CANN安装](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)

cann安装8.2.rc1版本，并source



## 环境搭建

```shell
conda create -n ling_lf python=3.10

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch_npu,metrics,deepspeed]"

pip install transformers==4.52.3
```



## 数据集

以LLaMA-Factory中的自我认知数据集为例，文件路径：LLaMA-Factory/data/identity.json，对其中的 name 和 author 进行替换，例如name换成 GLM4，author替换成 JYX



## 模型权重

从hugging face中下载模型权重

将模型权重放至 /home/ling/ling_mini_v2



## 训练配置

新建yaml文件，写入如下内容

```yaml
# model
model_name_or_path: /home/ling/ling_mini_v2
trust_remote_code: true

# method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

# dataset
dataset: identity
cutoff_len: 1024
template: bailing

# output
output_dir: /home/ling/output   # 输出路径
logging_steps: 1
save_steps: 5 
overwrite_output_dir: true
save_total_limit: 2

# train
per_device_train_batch_size: 1
learning_rate: 1.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
num_train_epochs: 1
seed: 1234
save_strategy: "no"
```

将训练配置保存为 ling_ds3.yaml

## 启动训练
在LLaMA Factory目录下运行如下命令启动训练

```shell
llamafactory-cli train ling_ds3.yaml
```