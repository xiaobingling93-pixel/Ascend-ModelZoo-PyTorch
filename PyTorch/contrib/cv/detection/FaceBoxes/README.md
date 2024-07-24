# FaceBoxes for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FaceBoxes是一款可以在cpu上实现实时，高准确率的目标检测模型。它包含快速消化卷积层和多尺度卷积层，快速消化卷积层用来解决CPU上的实时问题，多尺度卷积层用来提高目标在不同尺度下的检测性能。

- 参考实现：

  ```
  url=https://github.com/zisianw/FaceBoxes.PyTorch
  commit_id=9bc5811fe8c409a50c9f23c6a770674d609a2c3a
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |
  | PyTorch 1.11.0 | torchvision==0.12.0 |

  **表 2**  HDK支持表
  
  | HDK        | 版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | 驱动 | >=23.0.3 |
  | CANN | >=7.0.0 |

- 环境准备指导

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  pip install -r 1.11_requirements.txt # PyTorch1.11版本
  ```
  > **说明:** 只需执行一条对应的PyTorch版本依赖安装命令。
- 编译环境

  导入环境变量。
   ```bash
   source scripts/npu_set_env.sh
   git clone https://github.com/Levi0223/FDDB_Evaluation.git
   cd FDDB_Evaluation
   python3 setup.py build_ext --inplace
   mv ../convert.py ../split.py ../evaluate.py ./
   ```

## 准备数据集

   用户自行获取原始数据集，可选用的开源数据集包括WIDER_FACE等，将图片放在如下所示目录下（使用WIDER_train数据集进行训练,数据处理过程如下）。

   ```bash
   #$FaceBoxes_ROOT 为项目根目录
   cd $FaceBoxes_ROOT/data
   mkdir WIDER_FACE
   cd WIDER_FACE
   #传WIDER_train.zip至WIDER_FACE
   #下载文件格式转换脚本
   git clone https://github.com/akofman/wider-face-pascal-voc-annotations.git
   mv WIDER_train.zip wider-face-pascal-voc-annotations
   cd wider-face-pascal-voc-annotations
   unzip WIDER_train.zip
   ./convert.py -ap ./wider_face_split/wider_face_train_bbx_gt.txt -tp ./WIDER_train_annotations/ -ip ./WIDER_train/images/
   mv WIDER_train_annotations annotations
   mv annotations $FaceBoxes_ROOT/data/WIDER_FACE
   mv WIDER_train/images $FaceBoxes_ROOT/data/WIDER_FACE
   cd $FaceBoxes_ROOT/data/WIDER_FACE
   cp gen_train_data.py $FaceBoxes_ROOT/data/WIDER_FACE
   cd $FaceBoxes_ROOT/data/WIDER_FACE
   python gen_train_data.py
   ```

   以WiderFace数据集为例，数据集目录结构参考如下所示。

   ```
   data
    ├── WIDER_FACE
    │   ├── images
    |   │   ├──0--Parade
    |   │   ├── ...
    |   │   ├──38--Tennis
    │   |   ├── ...
    │   ├── annotations
    |   │   ├──0_Parade_marchingband_1_100.xml
    |   │   ├── ...
    |   │   ├──0_Parade_marchingband_1_6.xml
    │   |   ├── ...
    │   ├── img_list.txt
   ```

   > **说明：**
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./scripts/train_1p.sh  # 单卡精度

     bash ./scripts/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./scripts/train_8p.sh  # 8卡精度

     bash ./scripts/train_performance_8p.sh  # 8卡性能
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --training_dataset                  //数据集路径
   --dist_url                          //分布式地址
   --multiprocessing-distributed       //是否使用多卡训练
   --print-freq                        //使用频率
   --num_workers                       //加载数据进程数
   --world_size                        //使用卡数
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --device                            //使用设备
   --rank                              //使用卡排名
   多卡训练参数：
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```

   训练完成后，权重文件保存在weight路径下。

## 精度测评
使用FDDB数据集进行精度测试，需要下载的文件名称为：originalPics.tar.gz。
在data路径下，创建FDDB/images文件夹，将originalPics.tar.gz解压到images路径下，数据处理过程参考如下。
```
tar -zxvf originalPics.tar.gz -C images
cd $FaceBoxes_ROOT
cp fddb_data_gen.py data/FDDB
cp -r FDDB_Evaluation/grond_truth data/FDDB
cd data/FDDB
python fddb_data_gen.py
```

整理好的数据集如下。
```
   data
    ├── FDDB
    │   ├── images
    │   │   ├──2002
    │   │   ├──2003
    \   img_list.txt
```
修改test.sh中的train_epoch参数，选择需要参与精度测评的权重代数，执行test.sh生成测评数据。
```
bash test.sh
```
计算精度结果, 修改eval.sh中的train_epoch参数，修改成test.sh中的一致。
```
bash eval.sh
```
   > **说明：**
   >精度测试的过程只作为一种参考。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | -  |   1    |    O2     |      1.5      |
| 8p-竞品V | 0.944 | - |  300   |   O2     |      1.5      |
|  1p-NPU  |   -   | -  |   1    |    O2   |      1.5      |
|  8p-NPU  | 0.9396  | -  |  300   |    O2    |      1.5      |

# 版本说明

## 变更

2024.07.17：更新内容，重新发布。

2020.07.08：首次发布。

## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 [public_address_statement.md](./public_address_statement.md)。