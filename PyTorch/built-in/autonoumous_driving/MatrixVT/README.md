# MatrixVT for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [MatrixVT](#MatrixVT)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [训练任务](#训练任务) 
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

MatrixVT是一个基于Transformer结构的BEV 3D检测模型，没有定制化算子。针对目前BEV中更有优势的Lift-Splat类方法中关键模块（Vision Transformation），MatrixVT实现了非常优雅的优化，在保持模型性能（甚至略微提高）的同时，能大幅降低计算量和内存消耗。

## 支持任务列表
本仓已经支持以下模型任务类型

| 模型 | 任务列表 | 是否支持 |
|:---:|:---:|:----:|
|MatrixVT | 训练 |  ✔   |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/Megvii-BaseDetection/BEVDepth/
  commit_id=d78c7b58b10b9ada940462ba83ab24d99cae5833
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/autonoumous_driving/MatrixVT/
    ```

# MatrixVT

## 准备训练环境

### 安装模型环境

- 推荐参考[配套资源文档](https://www.hiascend.com/developer/download/commercial)使用最新的配套版本。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 24.1.RC3 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 24.1.RC3 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.0.RC3 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v6.0.rc3 </td>
    </tr>
  </table>

- 三方库依赖如下表所示。

  **表 2**  三方库依赖表

  |        三方库        |      支持版本      |
  |:-----------------:|:--------------:|
  |      PyTorch      |  2.1.0 |
  |    TorchVision    | 0.16.0 |
  |       NumPy       |     1.23.5     |
  | pytorch-lightning |     1.6.5      |
  |      mmdet3d      |    1.0.0rc4    |
  |       mmcv        |  1.7.1, 1.7.2  |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```shell
  pip install -r requirements.txt        # PyTorch 2.1版本
  ```

  手动安装指定三方库:  

- 安装mmcv==1.x（如果环境中有mmcv，请先卸载再执行以下步骤）。
  ```shell
  git clone -b 1.x https://github.com/open-mmlab/mmcv
  cd mmcv
  
  安装完成后将源码根目录下面code_for_change中的setup.py替换到mmcv中的setup.py
  
  MMCV_WITH_OPS=1 pip install -e . -v
  
  安装完成mmcv之后，进入{mmcv_install_path}/mmcv/ops/deform_conv.py文件夹，修改deform_conv.py文件，修改内容如下：
  导入torch_npu
  修改第56行，将torch.npu_deformable_conv2dbk修改为torch_npu.npu_deformable_conv2dbk
  
  修改modulated_deform_conv.py，修改内容如下：
  导入torch_npu
  修改第59行，将torch.npu_deformable_conv2d修改为torch_npu.npu_deformable_conv2d
    
  cd ../
  ```

- 安装mmdet3d==1.0.0rc4（如果环境中有mmdet，请先卸载再执行以下步骤）。
  ```shell
  pip install mmdet3d==1.0.0rc4
  ```
  

返回模型根目录
```shell
python setup.py develop
```
源码替换
```shell
pip show pytorch_lightning
```
找到pytorch_lightning的安装路径，然后将源码根目录下面code_for_change中的文件替换掉pytorch_lightning安装路径下对应的文件。
```shell
cp ./code_for_change/subprocess_script.py {pytorch_lightning_install_path}/strategies/launchers/subprocess_script.py
cp ./code_for_change/training_epoch_loop.py {pytorch_lightning_install_path}/loops/epoch/training_epoch_loop.py
cp ./code_for_change/types.py {pytorch_lightning_install_path}/utilities/types.py
cp ./code_for_change/accelerator_connector.py {pytorch_lightning_install_path}/trainer/connectors/accelerator_connector.py
```

### 准备数据集

   1. 请用户自行获取并解压nuScenes数据集，并将数据集的路径软链接到 `./data/`。
       ```
       ln -s [nuscenes root] ./data/
       ```

   2. 在源码根目录下进行数据集预处理。

       ```
       python scripts/gen_info.py
       python scripts/gen_depth_gt.py
       ```

      参考数据集结构如下：

      ```
       MatrixVT for PyTorch
       ├── data
       │   ├── nuScenes
       │   │   ├── maps
       │   │   ├── samples
       │   │   ├── sweeps
       │   │   ├── v1.0-test
       |   |   ├── v1.0-trainval
       ```
       
       > **说明：**  
       该数据集的训练过程脚本只作为一种参考示例。      


## 快速开始
### 训练任务

本任务主要提供**混精fp16**的**8卡**训练脚本。

#### 开始训练

  1. 进入源码根目录。

     ```
     cd /${模型文件夹名称}
     ```

  2. 运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡训练

     ```
     bash ./test/train_full_8p.sh # 8卡精度，混精fp16
     bash ./test/train_performance_8p.sh # 8卡性能，混精fp16
     ```

     > 注：当前配置下，不需要修改train_full_8p.sh中的ckpt路径，如果涉及到epoch的变化，请用户根据路径自行配置ckpt。

     模型训练脚本参数说明如下。
   
     ```
     matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema.py
     --seed                              // 随机种子
     --learning_rate                     // 学习率
     --max_epoch                         // 最大迭代回合数
     --amp_backend                       // 混精策略
     --gpus                              // 卡数
     --precision                         // 训练精度模式
     --batch_size_per_device             // 每张卡的批大小
     ```


#### 训练结果
##### 精度

| 芯片       | 卡数 | mAP   |  mATE  |  mASE  |  mAOE  |  mAVE  |  mAAE  |  NDS   | batch_size | AMP_Type | Torch_Version | 
|----------|:--:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:----------:|:--------:|:-------------:|
| 竞品A      | 8p | 0.3245 | 0.6546 | 0.2712 | 0.5572 | 0.8833 | 0.2288 | 0.4027 |     64     |   fp16   |      2.1      |
| Atlas 800T A2 | 8p | 0.3247 | 0.6555 | 0.2716 | 0.5251 | 0.8661 | 0.2316 | 0.4073 |     64     |   fp16   |      2.1      |

##### 性能

| 芯片       | 卡数 | FPS  | Each epoch time  | batch_size | AMP_Type | Torch_Version | 
|----------|:--:|:----:|:----:|:----:|:--------:|:-------------:|
| 竞品A      | 8p |  56  |  0.65 h  | 64 |   fp16   |      2.1      |
| Atlas 800T A2 | 8p | 45 |   0.81 h  | 64|  fp16    |      2.1      |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md 及 README.md

# 变更说明

2023.11.09：首次发布。

# FAQ

1. 报错scikit_learning.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block的问题，解决方案为：

   ```
   # 手动导入环境变量
   export LD_PRELOAD={libgomp-d22c30c5.so.1.0.0_path}/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
   ```
2. 如果使用pip安装mmdet3d失败，请采用源码安装的方式。源码下载地址为：
   ```
   https://github.com/open-mmlab/mmdetection3d/releases/tag/v1.0.0rc4
   ```
   请下载链接中的源码压缩包，自行手动安装。同时修改mmdet3d的源码，进入mmdet3d的安装路径，修改内容有:
   ```
   1) cd /${mmdet3d-1.0.0rc4的安装路径}/mmdet3d/
     将__init__.py中的文件第41行mmcv_maximum_version = '1.7.0'修改为mmcv_maximum_version = '1.7.2'
   2) cd /${mmdet3d-1.0.0rc4的安装路径}/requirements/
     删除runtime.txt中的numba==0.53.0这一行。
   ```
   然后执行：
   ```
   cd /${mmdet3d-1.0.0rc4的安装路径}/
   pip install -v -e .
   ```

