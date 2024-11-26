





# YOLOX for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

MMDetection 是一个基于 PyTorch 的目标检测开源工具箱。计算机视觉基础库 [MMCV](https://github.com/open-mmlab/mmcv)，MMCV 是 MMDetection 的主要依赖。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=3e36d5cfd4fe7c550b4c3493360fd369b858b1dc
  
  配置文件：
  url=https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/yolox_m_8x8_300e_coco.py
  commit_id=2bdb1670e1f78930b0cd959263ed667ecada954d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/dev/cv/detection
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动   | [1.0.25.alpha](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.0.alpha001&driver=1.0.25.alpha) |
  | CANN       | [8.0.0.alpha001](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.alpha001) |
  | Ascend Extension for PyTorch | [2.1.0](https://gitee.com/ascend/pytorch/tree/v2.1.0/) |
  | mxDriving | [6.0.0-RC2](https://gitee.com/ascend/mxDriving/tree/branch_v6.0.0-RC2/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  1. 源码编译安装 mmcv 1.x
     ```
      git clone -b 1.x https://github.com/open-mmlab/mmcv.git
      cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/distributed.py
      cp -f mmcv_need/swish.py mmcv/mmcv/cnn/bricks/swish.py
      cp -f mmcv_need/runtime.txt mmcv/requirements/runtime.txt
      cd mmcv
      pip install -r requirements/runtime.txt
      MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
      MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
     ```
  2. 源码安装 mmdet 2.25.3
     ```
     cd ModelZoo-PyTorch/PyTorch/dev/cv/detection/YOLOX_ID2833_for_PyTorch
     pip install -e .
     ```
  3. 安装其他依赖
     ```
     pip install -r requirements.txt
     ```
  4. 安装mxDriving加速库，并export环境变量：
     ```
     export ASCEND_CUSTOM_OPP_PATH=xxx/site-packages/mx_driving/packages/vendors/customize/
     export LD_LIBRARY_PATH=xxx/site-packages/mx_driving/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
     ```
   【注意】当前版本配套mxDriving RC3及以上版本，历史mxDriving版本需要model仓代码回退到git reset --hard 91ac141ecfe5872f4835eef6aa4662f46ede80c3
## 准备数据集

1. 获取数据集。

   使用coco2017数据集。
   准备好数据集后放到 ./dataset 目录下

   ```
   ├── coco2017
         ├── annotations               
         	├── instances_train2017.json
         	├── instances_val2017.json ...
         ├── train2017
         	├── 000000******.jpg ...
         ├── val2017
         	├── 000000******.jpg ...
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理
   
    - 本模型不涉及

## 获取预训练模型（可选）

- 本模型不涉及

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   单机单卡训练

   ```
   cd test
   bash train_full_1p.sh --data_path=xx/xx/coco2017
   或
   bash train_performance_1p.sh --data_path=xx/xx/coco2017
   ```

   

   单机8卡训练

   ```
   
   cd test
      bash train_full_8p.sh --data_path=xx/xx/coco2017
      或
      bash train_performance_8p.sh --data_path=xx/xx/coco2017
   ```
   

训练完成后，pth文件保存在./work_dirs下面，并输出模型训练精度和性能信息。

   注意：train_full_8p.sh为通用的8P全量精度训练脚本，train_full_8p_resume.sh 在train_full_8p.sh的基础上新增了异常退出后复跑机制，注意是规避三方组件异常退出等，确保训练可以完成。



​	3、验证模型

​		可使用提供的 test_1p.sh或 test_8p.sh对上述训练出的pth文件进行推理验证。

​		注意：1p与1p的pth对应，8p与8p的pth对应。

​		

```
   cd test
   bash test_1p.sh --data_path=xx/xx/coco2017 --pth_path=xx/work_dirs/yoloxxx/*.pth
   或
   bash test_8p.sh --data_path=xx/xx/coco2017 --pth_path=xx/work_dirs/yoloxxx/*.pth
```

​		



# 训练结果展示

**表 2**  训练结果展示表

| NAME     | mAp(Iou=0.50:.95) |  FPS | Steps     |
| -------  | :---:  | ---: | :----:    |
| 8p-NPU   | 0.461 | 238 | 4,437,600 |
| 8p-竞品A |       0.462       | 322 | 4,437,600 |



# 版本说明

## 变更

2022.11.29：首次发布
2024.08.20: 性能优化

## 已知问题

**_当前发行版本中存在的问题描述。_**

1、全量训练下，配置data.persistent_workers=False，mmdetection存在精度问题： https://github.com/open-mmlab/mmdetection/issues/9530 ，后续在三方组件修复此问题后，刷新同步。

2、全量训练下，配置data.persistent_workers=True，训练过程概率性内存异常，需要刷新Torch版本（预计2023-2 BugFix），可选用train_full_8p_resume.sh 临时规避。




# 公网地址说明
代码涉及公网地址参考 public_address_statement.md




