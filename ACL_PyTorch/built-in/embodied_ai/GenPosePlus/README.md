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
| Ascend Extension PyTorch                                        | 2.1.0.post17 | -                                                                                                     |
| 说明：支持Atlas 300I DUO/ 310P RC | \           | \                                                                                                     |



## 环境准备

- 安装conda环境（略）

使用conda创建虚拟环境并激活：

```bash
conda create -n genpose2 python==3.10.14
conda activate genpose2
```
- 安装推理工具ais_bench

ais_bench安装请参考[ais_bench安装指导](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)


## 获取源码并处理
获取源码，切到指定commit，激活必要环境，安装必要依赖：

```bash
# 获取源码
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/GenPosePlus
export PYTHONPATH=$PWD:$PYTHONPATH
git clone https://github.com/Omni6DPose/GenPose2.git
cd GenPose2
export PYTHONPATH=$PWD:$PYTHONPATH

# 切换至指定commit
git reset d0993c0

#应用补丁，注意，此命令只能执行一次，再次执行会报错（不影响后续使用）
git apply ../diff.patch
export GPPATH=$PWD

# 激活必要环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


#安装相关依赖：
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

- 在数据集下载页面ROPE同级目录找到 `Meta/`路径下内容并复制至 `$GPPATH/configs/` 路径，组织如下:

``` bash
GenPose2
└──configs
   ├── obj_meta.json
   ├── real_obj_meta.json
   └── config.py
```

- 模型可在如下连接找到 [checkpoints](https://www.dropbox.com/scl/fo/x87lhf7sygjm1gasz153g/AIHBlaGMjhfyW1bKrDe61R4?rlkey=y1f6dqdi40tzcgepccthayudp&st=1sbkxbzf&dl=0). 请下载至如下路径 `$GPPATH/results` 并组织如下:

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
### Single模式评测

该模式会将数据集下所有数据视为没有关联，随机每一个数据的初始值。

- 转换pth模型为onnx模型
```bash
# 进入项目目录
cd $GPPATH
# 执行转换脚本，运行需要一些时间
python ../export_all_onnx.py
```
- 转换onnx模型为om模型
```bash
# 执行转换脚本，运行需要一些时间
python ../export_all_om.py
# 可跟参数 --soc_version，缺省为Ascend310P3
```

执行如下命令进行评测
```bash
bash scripts/eval_single_om.sh
```
### Tracking模式评测

该模式会将数据集下所有数据视为有连续关联，非首数据的初始值会基于上一个数据。

- 转换pth模型为onnx模型
```bash
# 进入项目目录
cd $GPPATH
# 执行转换脚本，运行需要一些时间
python ../export_all_onnx.py --batch_size 4
```
- 转换onnx模型为om模型
```bash
# 执行转换脚本，运行需要一些时间
python ../export_all_om.py --batch_size 4
# 可跟参数 --soc_version，缺省为Ascend310P3
```

执行如下命令进行评测
```bash
bash scripts/eval_tracking_om.sh
```

**注意：** 两种方式，非首次推理必须删除上一次缓存，否则所得结果为上一次缓存。
参考删除缓存命令
```bash
rm -rf $GPPATH/results/evaluation_results
```


模型推理性能精度结果：
| 芯片 | 推理方式 |  batchsize  |     iou_mean   |   参考性能(ms/sample)     |
|------|-----|--------|-------------|-------------|
|  300I DUO| single | 16  |        0.3073     | 652.20 |
|  300I DUO| tracking | 4  |        0.3038     | 458.37 |
|  310P RC| single | 16  |        0.3073     | 803.23 |
|  310P RC| tracking | 4  |        0.3036     | 488.15 |


