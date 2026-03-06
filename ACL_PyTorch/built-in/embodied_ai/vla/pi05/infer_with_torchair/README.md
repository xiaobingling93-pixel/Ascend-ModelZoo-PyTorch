# pi05(TorchAir)-推理指导

## 概述
LeRobot 是一套基于 PyTorch 构建的面向真实世界机器人应用的工具集，核心目标是降低机器人领域的研发门槛，让开发者均可参与共享数据集、预训练模型的建设。该指导基于LeRobot框架，采用TorchAir加速路线部署VLA模型--π0.5。

## 插件与驱动准备

- 该模型需要以下插件与驱动

  | 配套                                                            | 版本          | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    |-------------| ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 24.1.RC3    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 8.3.RC1     | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                          | 3.11        | -                                                                                                     |
  | PyTorch                                                         | 2.7.1       | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.7.1 | -                                                                                                     |
  | 说明：支持Atlas 300I DUO/Atlas 300I Pro，不支持Atlas 800I A2 | \           | \                                                                                                     |


## 获取本仓源码
```
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/vla/pi05/infer_with_torchair
```

## 环境准备

* 获取LeRobot源码，切到指定commit并安装相关依赖：
```
git clone -b v0.4.3 https://github.com/huggingface/lerobot.git
cd lerobot
# 安装LeRobot基础依赖
pip install -e .
# 安装Libero仿真环境依赖
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install -e ".[libero]"
# 安装π0.5相关依赖
pip install -e ".[pi]"
# 叠加patch
cd ../patches
python3 apply_patch.py
cd ..
```

* 安装OpenGL/EGL相关底层依赖库
```
apt install -y libegl1-mesa libegl1-mesa-dev libopengl0 libgl1-mesa-glx libgl1-mesa-dev
```


* 安装requirements：
  `pip3 install -r requirements.txt`

* 下载模型权重：
```
mkdir weight
cd weight
```
  * pi05_libero_finetuned：[下载链接](https://modelscope.cn/models/lerobot/pi05_libero_finetuned/files)
  * paligemma-3b-pt-224 tokenizer：[下载链接](https://modelscope.cn/models/AI-ModelScope/paligemma-3b-pt-224/files)
```
cd ..
```

## 文件目录结构
文件目录结构大致如下：

```text
📁 pi05/
├── 📁 infer_with_torchair/
|   |── 📁 lerobot
|   |── 📁 patches
|   |   |── 📄 apply_patch.py
|   |   |── 📄 lerobot_diff.patch
|   |   └── 📄 modeling_siglip.patch
|   |── 📄 infer.py
|   |── 📄 README.md
|   └── 📄 requirements.txt
```

## 模型推理
  ```
  # 1. 指定使用NPU ID，默认为0
  export ASCEND_RT_VISIBLE_DEVICES=0
  # 2. 执行推理脚本，输出单个样本推理耗时，首次执行时会自动下载libero数据集。
  python3 infer.py --pi05_model_path=/path/to/pi05_libero_finetuned --tokenizer_path=/path/to/paligemma-3b-pt-224
  ```
  infer.py推理参数：
  * --model_path: pi05模型权重路径，默认为"./weight/pi05_libero_finetuned"
  * --tokenizer_path: paligemma-3b-pt-224 tokenizer配置文件路径，默认为"./weight/paligemma-3b-pt-224"
  * --warmup: warm up次数，默认为3，首次warm up时编译成图
  * --episode_index: 选择数据集样本的index

## 性能数据
  infer.py取libero数据集中的一个frame进行推理，性能如下

  | 模型                     | 芯片 | 数据集 | 单次推理耗时
  |-------------------------|------|----------|
  |π0.5    |  300I DUO | libero       | 620ms |

## 精度测试
  执行以下命令来对libero数据集做仿真精度测试，首次执行时会下载
  ```
  python3 -m lerobot.scripts.lerobot_eval --policy.path=/path/to/pi05_libero_finetuned --env.type=libero --env.task=libero_object --eval.n_episodes=1 --eval.batch_size=1 --output_dir=my_eval_output --tokenizer_path=/path/to/paligemma-3b-pt-224
  ```
  lerobot_eval.py推理参数：
  * --policy.path: pi05模型权重路径
  * --tokenizer_path: paligemma-3b-pt-224 tokenizer配置文件路径
  * --env.type: 仿真环境类型
  * --env.task: 任务类型
  * --eval.n_episodes: 每个环境下执行次数
  * --output_dir: 渲染的视频和输出结果都会保存在该目录下 

  | 模型                     | 芯片 | libero object成功率         | 竞品libero object成功率|
  |-------------------------|------|-------------| ------ |
  |π0.5      |  300I DUO|  100 | 98.2|

## FAQ
1. 安装依赖时报CMake Error: CMake Error at CmakeLists.txt:1(cmake_minimum_required)....
export CMAKE最低版本：
  ```
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  ```
  重新执行安装依赖的命令

2. 运行run_eval.sh报错 ValueError: numpy.dtype size changed, may indicate binary incompatibility.
numpy与panda版本不匹配，查看numpy版本，重新安装numpy: pip install numpy==1.26.2
