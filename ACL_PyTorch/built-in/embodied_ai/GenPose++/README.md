# GenPose++ 推理指导

## 概述
GenPose++ 采用分段点云和裁剪后的 RGB 图像作为输入，利用 PointNet++ 提取物体的几何特征；同时借助预训练的 2D 基础骨干网络 DINO v2 提取通用语义特征。随后，将这些特征融合，作为扩散模型的条件，生成物体姿态候选及其对应的能量。最后，针对具有非连续对称性（如盒子）的物体，通过聚类解决姿态多模态分布带来的聚集问题，从而有效完成姿态估计。

## 插件与驱动准备

- 该模型需要以下插件与驱动

| 配套                                                            | 版本          | 环境准备指导                                                                                          |
| ------------------------------------------------------------    |-------------| ------------------------------------------------------------                                          |
| 固件与驱动                                                       | 24.1.RC3    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 8.3.RC1     | 包含kernels包和toolkit包                                                                                                   |
| Python                                                          | 3.10.14        | -                                                                                                     |
| PyTorch                                                         | 2.1.0       | -                                                                                                     |
| Ascend Extension PyTorch                                        | 2.1.0 | -                                                                                                     |
| 说明：支持Atlas 300I DUO | \           | \                                                                                                     |


## 获取本仓源码
```bash
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/GenPose++
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 环境准备

* 获取源码，切到指定commit：
```bash
git clone https://github.com/Omni6DPose/GenPose2.git
cd GenPose2
git reset d0993c0
git apply ../diff.patch
export ROOT=$PWD
```

安装conda环境（略）

使用conda创建虚拟环境并激活：

```bash
conda create -n genpose2 python==3.10.14
conda activate genpose2
```

激活必要环境
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

安装相关依赖：

安装相关依赖：


``` bash
pip install -r requirements.txt 
```

* 安装requirements：

```bash
pip3 install -r requirements.txt
pip3 install -r ../requirements.txt
```


### 下载配置文件与模型

请按照 [Omni6DPoseAPI](https://github.com/Omni6DPose/Omni6DPoseAPI)页面上的说明，下载并整理 Omni6DPose 数据集中的`ROPE`分类中的`000000`号数据集，并在项目目录中组织如下

```bash
omni6dpose-000000
└──ROPE
   └── 000000
	  ├── 000000_color.png
	  ├── 000000_depth.exr
	  ├── 000000_mask.exr
	  ├── 000000_mask_sam.npz
	  ├── 000000_meta.json
	  ...
	  └── 000945_meta.json
```

- 在数据集下载页面ROPE同级目录找到 `Meta/`路径下内容并复制至 `$ROOT/configs/` 路径，组织如下:

``` bash
GenPose2
└──configs
   ├── obj_meta.json
   ├── real_obj_meta.json
   └── config.py
```

- 模型可在如下连接找到 [checkpoints](https://www.dropbox.com/scl/fo/x87lhf7sygjm1gasz153g/AIHBlaGMjhfyW1bKrDe61R4?rlkey=y1f6dqdi40tzcgepccthayudp&st=1sbkxbzf&dl=0). 请下载至如下路径 `$ROOT/results` 并组织如下:

``` bash
GenPose2
└──results
   └── ckpts
       ├── ScoreNet
       │   └── scorenet.pth
       ├── EnergyNet
       │   └── energynet.pth
       └── ScaleNet
           └── scalenet.pth
```


## 推理与评测

执行如下命令进行评测
```
bash scripts/eval_single.sh
```
注：如为310P RC设备，需将eval_single.sh中bs修改为16.


模型推理性能精度结果：
| 芯片 |   batchsize  |     iou_mean   |   性能(ms/sample)     |
|------|-------------|-------------|-------------|
|  300I DUO| 32  |        0.2928     | 2973.17 |
|  310P RC| 16  |        0.2959     | 3861.52 |


