# Chinese-Clip(TorchAir)-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******

# 概述
Chinese-Clip为CLIP模型的中文版本，使用大规模中文数据进行训练，并针对中文领域数据以及在中文数据上实现更好的效果做了优化。CLIP是2021年OpenAI提出的基于图文对比学习的多模态预训练模型，具备强大的zero-shot迁移能力。

- 版本说明：
  ```
  url=https://github.com/OFA-Sys/Chinese-CLIP/
  commit_id=85f3fa3639e207d0b76f69a401105cad5d509593
  model_name=Chinese-Clip
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            | 版本           | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    |--------------| ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 24.0.0       | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 8.0.0        | -                                                                                                   |
  | Python                                                          | 3.10.16      | -                                                                                                     |
  | PyTorch                                                         | 2.1.0        | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.1.0.post10 | -                                                                                                     |
  | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \            | \                                                                                                     |


# 快速上手

## 获取源码

1. 获取模型仓源码  
   ```bash
   git clone https://github.com/OFA-Sys/Chinese-CLIP
   cd Chinese-CLIP
   git reset --hard 85f3fa36
   ```
   
2. 安装依赖  
   ```bash
   pip3 install -r ../requirements.txt
   ```

## 模型推理
1. 执行patch文件
    ```bash
    git apply adapt-torchair.patch
    ```

2. 移动推理py文件到源码目录内
    ```bash
    cp ../infer_air.py ./
    ```
   
3. 下载模型权重

    下载[开源权重](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt)并上传到服务器，或通过wget直接下载权重。
    其他权重链接参考[模型仓](https://github.com/OFA-Sys/Chinese-CLIP)下载链接
    ```bash
    wget https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt
    ```
   
4. 执行推理命令

    ```bash
    python3 infer_air.py \
       --model_root_path=./ \
       --model_type=ViT-B-16 \
       --image=examples/pokemon.jpeg \
       --text="杰尼龟" \
       --length=64 \
       --batch_size=16 \
       --device=0 \
       --loop=10
    ```
    - 参数说明
      - model_root_path：模型权重所在文件夹路径
      - model_type：模型规模，默认为ViT-B-16
      - image：输入图片，默认为examples/pokemon.jpeg
      - text：输入文本，默认为"杰尼龟"
      - length：Tokenizer输出长度，默认为64
      - batch_size：模型输入batch_size，默认为16
      - device：npu芯片id，默认为0
      - loop：性能测试的循环次数，默认为10
  
    推理脚本以计算图片文本相似度为例，推理后将打屏推理结果和模型性能

# 模型推理性能&精度
以ViT-B-16模型，bs=16，length=64为例

| 模型     | 硬件 | 余弦相似度 | 端到端性能        |
|--------|------|-----|--------------|
| visual |800I A2| 1 | 1748 image/s |
| bert   |800I A2| 1 | 3614 text/s  |

