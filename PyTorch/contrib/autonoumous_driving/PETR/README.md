# PETR for Pytorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [PETR](#PETR)   
    -   [准备训练环境](#准备训练环境)
          - [安装模型环境](#安装模型环境)
          - [安装依赖](#安装依赖)
          - [安装昇腾环境](#安装昇腾环境)
          - [准备数据集](#准备数据集)
          - [获取预训练权重](#获取预训练权重)
    -   [快速开始](#快速开始)
          - [支持单机8卡训练](#支持单机8卡训练)
          - [微调任务](#微调任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

PETR开发了用于多视图3D对象检测的位置嵌入变换(PETR)。PETR将3D坐标的位置信息编码为图像特征，生成3D位置感知特征。对象查询可以感知3D位置感知特征，并进行端到端的对象检测。
本仓库主要将PETR模型迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|  模型  |    任务列表    | 是否支持 |
|:----:|:----------:|:-----:|
| PETR |    训练FP16     | ✔ |
| PETR |    推理    | ✔ |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/megvii-research/PETR
  commit_id=f7525f93467a33707ef401c587a52d5e7b34de74
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/autonoumous_driving
  ```


# PETR
## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  | 三方库          | 支持版本 |
  | :--------:     | :------: |
  | PyTorch        | 1.11.0   |
  | mmcv           | 1.x      |
  | mmdet          | 2.28.2   | 
  | mmsegmentation | 0.30.0   |
  | mmdet3d        | 1.0.0rc7 |


### 安装依赖
1. 源码编译安装 mmcv 1.x（如果环境中有mmcv，请先卸载再执行以下步骤）
  ```shell
  git clone https://github.com/open-mmlab/mmcv.git
  cd mmcv
  git checkout 1.x
  # 代码标签: commit ed4303ea95569a01dcb253074e62cdcc945ff2d7
  git checkout ed4303ea95569a01dcb253074e62cdcc945ff2d7
  # 拷贝 mmcv.patch至mmcv源码目录目录下
  cp third/mmcv.patch ${work_dir}/mmcv
  cd mmcv
  git apply mmcv.patch
  #通过git diff 查看
  #编译安装mmcv
  MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
  ```
2. 安装mmdet==2.28.2
  ```shell
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
git checkout 2.x
  # 代码标签: commit e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1
  git checkout e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1
  pip install -e .
  ``` 

3. 源码安装mmsegmentation==0.30.0
  ```shell
  git clone https://github.com/open-mmlab/mmsegmentation.git
  cd mmsegmentation
  git checkout 0.x
  # 代码标签: commit f67ef9c128eb2b643beaed8eb518c9fa09eb0912
  git checkout f67ef9c128eb2b643beaed8eb518c9fa09eb0912
  pip install -e .
```
4. 源码安装mmdet3d==1.0.0rc7（如果环境中有mmdet，请先卸载再执行以下步骤）
  ```shell
  git clone https://github.com/open-mmlab/mmdetection3d.git
  cd mmdetection3d
git checkout 1.0
  # 代码标签: commit c0c378f2154238a65446f7e72481a2025df4bb4d
  git checkout c0c378f2154238a65446f7e72481a2025df4bb4d
  pip install -e .
  ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

  | 软件类型   | 支持版本  |
  | :--------: | :-------------: |
  | FrameworkPTAdapter | 在研版本 |
  | CANN | 在研版本 |
  | 昇腾NPU固件 | 在研版本 | 
  | 昇腾NPU驱动 | 在研版本 |

  

### 准备数据集

#### 训练数据集准备

**Step 0.** 请用户自行到nuScenes官网下载数据集，并按petr github指导进行预处理。

**Step 1.** 将数据集的路径软链接到 ${work_dir}/PETR/data/。

```
cd ${work_dir_work_dir}/PETR
ln -s [nuscenes root] ./data/nuscenes
```
数据集结构如下所示。
```
PETR for PyTorch
├── data
│   ├── nuScenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

### 获取预训练权重

请按照github issue下载petr backbone训练权重resnet50_msra-5891d200.pth  到 ${work_dir}/PETR/ckpts目录下。

## 快速开始

### 支持单机8卡训练
**Step 1.** 进入源码根目录。

```
cd /${container_work_dir}/PETR
```

**Step 2.**  运行训练脚本。

```
当前配置下，不需要修改train_full_8p.sh中的ckpt路径，如果涉及到epoch的变化，请用户根据路径自行配置ckpt。

bash ./test/train_full_8p.sh # 8卡训练. fp16
```
   
#### 训练结果


**表 3**  精度结果展示表

| Exp    |  mAP   |  mATE  |  mASE  |  mAOE  |  mAVE  |  mAAE  |  NDS   |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 竞品A_8p | 0.3052 | 0.8441 | 0.2827 | 0.6225 | 0.9824 | 0.2474 | 0.3547 |
| Atlas 800T A2_8p | 0.3047 | 0.8538 | 0.2818 | 0.6231 | 0.9579 | 0.2157 | 0.3591 |

**性能**

batch_size=14性能测试
```
samples_per_gpu=14,
workers_per_gpu=12,
```
**表 4**  性能结果展示表
| Exp    | FPS  | Each epoch time  |
|--------|:----:|:----------------:|
| 竞品A_8p |  46  |      0.169 h      |
| Atlas 800T A2_8p |  33  |      0.238 h      |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.3.18：petr fp16训练任务首次发布。

# FAQ



