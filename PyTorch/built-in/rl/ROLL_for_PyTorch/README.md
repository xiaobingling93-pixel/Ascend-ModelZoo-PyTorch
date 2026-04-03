# ROLL x 昇腾

最后更新时间：2026-04-03。

我们已在 ROLL 中添加了对华为昇腾设备的支持。

## 硬件支持 

Atlas 900 A3 PODc

## 安装

### 基础环境配置

| 软件 | 版本 |
| -------- |---------|
| Python   | 3.11    |
| CANN     | 8.5.1   |

### 创建 Conda 环境

在 Miniconda 中使用以下命令创建新的 conda 环境：

```
conda create --name roll python=3.11
conda activate roll
```

### 安装 torch & torch_npu

要在 ROLL 中使用 torch 和 torch_npu，请使用以下命令进行安装：

```
# 使用仅 CPU 版本的 torch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 torch_npu 2.9.0
pip install torch_npu==2.9.0
```

### 安装 vllm & vllm-ascend

要在 ROLL 中使用 vllm，请按以下步骤编译安装 vllm 和 vllm-ascend：

```
# vllm
git clone -b v0.13.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt

VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend
git clone -b v0.13.0 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

pip install -e .
cd ..
```

或者，您也可以从预构建的 wheel 包安装 `vllm` 和 `vllm-ascend`：
```
# 安装 vllm-project/vllm。最新支持的版本是 v0.13.0
pip install vllm==0.13.0

# 从 pypi 安装 vllm-project/vllm-ascend
pip install vllm-ascend==0.13.0
```

### 安装 ROLL

```
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt
pip install -e .
pip install deepspeed==0.16.4
pip install tensorboard
cd ..
```

### 其他第三方库

| 软件                    | 说明   |
| --------------------------- | ------------- |
| transformers                | >= v4.57.6    |
| flash_attn                  | 不支持 |
| transformer-engine[pytorch] | 不支持 |

1. `transformers` v4.57.6 支持启用 `--flash_attention_2`。
2. 目前不支持 `flash_attn` 加速。
3. 目前不支持 `transformer-engine[pytorch]`。

```
pip install transformers==4.57.6
```

## 快速开始：单节点部署

在正式使用之前，我们建议先测试单节点流程以验证您的环境和安装是否正确。
由于目前还不支持 Megatron-LM 训练，请先在相关文件中将 `strategy_args` 更改为使用 `deepspeed` 选项。

**注意：** 目前 NPU 上不支持 colocated 模式。您需要修改 `device_mapping` 以确保训练和推理在不同的卡上进行。


使用配置文件运行 rlvr 流程，将`qwen3_8b_rlvr_deepspeed.yaml`复制到xx文件架下，运行：

```
# 确保您在 ROLL 项目的根目录下

python examples/start_agentic_pipeline.py \
        --config_path qwen2.5-7B-rlvr-offpolicy \
        --config_name qwen3_8b_rlvr_deepspeed
```

- `--config_path` – 包含 YAML 配置文件的目录。
- `--config_name` – 文件名（不含 `.yaml` 扩展名）。

