# Kolors模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)
  
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

参考实现：
```bash
# Kolors
https://huggingface.co/Kwai-Kolors/Kolors
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | prompt    |  batch x 77 | STRING |  ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batch x 3 x 896 x 1408 | FLOAT32  | NCHW          |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.x | -                                                            |
   | torch| 2.1.0  | -                                                            |

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

0. 下载仓库到本地。
   ```bash
   git clone https://modelers.cn/MindIE/Kolors.git
   ```

1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt

   # 若要使用hpsv2验证精度, 则还需要按照以下步骤安装hpsv2
   git clone https://github.com/tgxs002/HPSv2.git
   cd HPSv2
   pip3 install -e .
   ```

2. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```

3. 安装gcc、g++

   ```shell
   # 若环境镜像中没有gcc、g++，请用户自行安装
   yum install gcc
   yum install g++
   # 导入头文件路径
   export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息生成图片，无需数据集。


## 模型推理<a name="section741711594517"></a>

1. 获取权重（可选）

    可提前下载权重，放到代码同级目录下，以避免执行后面步骤时可能会出现下载失败。

    ```bash
    # 需要使用 git-lfs (https://git-lfs.com)
    git lfs install
    
    # Kolors
    git clone https://huggingface.co/Kwai-Kolors/Kolors
    ```

2. 开始推理验证。
                   
    1. 开启cpu高性能模式
    ```bash
    echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    ```

    2. 执行推理脚本。
    ```bash
    # 使用上一步下载的权重
    model_base="./Kolors/"
    ```

    
    执行命令
    ```bash
    # 单卡推理
    python3 infer.py \
        --path=${model_base} \
        --prompt_file="./prompts/prompts.txt" \
        --height=1024 \
        --width=1024 \
        --output_dir="./images" \
        --steps=50 \
        --seed=65535 \
        --device_id=0 \
        --cache_method="agb_cache"
    ```
    参数说明：
    - --path: 模型权重路径
    - --prompt_file: 输入的prompt文件
    - --height: 生成图片的高
    - --width: 生成图片的宽
    - --output_dir: 生成图片的保存路径
    - --steps: 推理步数
    - --seed: 随机种子
    - --device_id: 推理设备ID
    - --cache_method: cache策略选择,支持配置"agb_cache"
    
    执行命令
    ```bash
    # 双卡推理
    ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2025 --nproc_per_node=2 infer.py \
        --path=${model_base} \
        --prompt_file="./prompts/prompts.txt" \
        --height=1024 \
        --width=1024 \
        --output_dir="./images" \
        --steps=50 \
        --seed=65535 \
        --cache_method="agb_cache" \
        --use_parallel

    ```
    参数说明：
    - --master_port: master节点的端口号，同于通信
    - --nproc_per_node: 一个节点中显卡的数量
    - --use_parallel: 开启双卡并行推理

  
3. 模型性能

参考性能结果:
| 800I A2 32G | 分辨率 | 迭代次数 | 单卡推理 | 双卡推理|
|-------|------|------|--------|-----------|
|无损优化| 1024x1024 | 50 | 8.28s | 6.20s |
|算法优化| 1024x1024 | 50 | 5.79s | 4.27s |

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。