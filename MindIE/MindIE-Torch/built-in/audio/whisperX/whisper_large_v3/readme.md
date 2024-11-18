# Whisper-large-v3模型-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)

# 概述
使用mindietorch部署whisper-large-v3模型


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套   | 版本        | 环境准备指导 |
  |-----------| ------- | ------------ |
  | Python | 3.10.13   | -            |
  | torch  | 2.1.0+cpu | -            |
  | torch_audio  | 2.1.0+cpu | -            |
  | CANN   | 8.0.RC3   | -            |
  | MindIE | 1.0.RC3   | -       |

# 快速上手
## 获取源码

1. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```


2. 模型权重下载路径:
    ```bash
    https://huggingface.co/openai/whisper-large-v3/tree/main
    ```
    将权重文件存放至当前目录下的model_path文件夹，请先创建改文件夹。
    

5. 安装依赖
    ```
    pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    pip3 install nltk
    pip3 install librosa
    pip3 install transformers==4.36.0
    pip3 install numpy==1.24.0
    ```

## 模型推理
1. 设置mindie内存池上限为32，执行如下命令设置环境变量。内存池设置过小，内存重复申请和释放会影响性能。
    ```
    export TORCH_AIE_NPU_CACHE_MAX_SIZE=32
    ```

2. 模型编译和推理
   ```
    python3 compile_whisper.py \
    -model_path ./model_path \
    -bs 16 \
    -save_path ./compiled_models \
    -soc_version *
    ```
    参数说明：
      - -model_path：预训练模型路径,必选。
      - -bs：batch_size, 默认值为16， 可选。
      - -save_path: 编译好的模型的保存文件，必选。
      - -device_id: 选在模型运行的卡编号，默认值0，可选。
      - -soc_version: 芯片类型,必选。
      - -hardware: 机器型号，默认值800IA2，可选["300IPro", "800IA2"]。
    约束说明：
        1. 当前暂不支持动态batch，batch_size改变后，需要重新编图。
        2. 支持的hardware类型为"300IPro"或"800IA2"。
        3. 芯片类型需要用户在环境上查询得到。
        如果无法确定当前设备的soc_version，则在安装NPU驱动包的服务器执行npu-smi info命令进行查询，
        在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。