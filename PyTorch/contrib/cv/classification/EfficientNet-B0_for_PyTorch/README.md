# EfficientNet-B0 for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)



# 概述

## 简述

EfficientNet是一个新的卷积网络家族，与之前的模型相比，具有更快的训练速度和更好的参数效率。
该模型通过一组固定的缩放系数统一缩放这在网络深度，网络宽度，分辨率这三方面有明显优势。
在EfficientNet中，这些特性是按更有原则的方式扩展的，也就是说，一切都是逐渐增加的。

- 参考实现：

  ```
  url=https://github.com/lukemelas/EfficientNet-PyTorch
  commit_id=7e8b0d312162f335785fb5dcfa1df29a75a1783a
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

# 准备训练环境

## 准备环境

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

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集imagenet2012。数据集下载之后的相关压缩包如下，其中`ILSVRC2012_img_train.tar`为训练数据集，`ILSVRC2012_img_val.tar`为测试数据集
   ```
   ILSVRC2012_img_train.tar
   ILSVRC2012_img_val.tar
   ```
2. 数据集预处理。
   将数据集组织方式处理为符合程序输入。
- 训练数据集预处理
   ```
   # step 1
   # 创建train文件夹，将tar转移到train文件中，并cd到train文件夹
   mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    
   # step 2
   # 解压 train压缩包并删除train压缩包
   # tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    
   # step 3
   # 解压1000个类别压缩包并创建对应的子文件。
   # find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
   ```  
- 测试数据集预处理

   ```
   # Step 1
   #创建val文件夹，将val.tar移动到val文件中，cd到val文件，解压
   mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    
   # Step 2
   # 将源码工程tools目录下的valprep.sh拷贝到val目录
  
   # Step 3
   # 重新分类
   valprep.sh
   ```
- 整理后数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
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
     bash ./test/train_full_1p.sh --data_path=real_data_path  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # 单卡性能
     ```
   
- 单机8卡训练

     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  # 8卡精度

     bash ./test/train_performance_8p.sh --data_path=real_data_path  # 8卡性能 
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data                              //数据集路径
   --arch                              //使用模型，默认：efficientnet-b0
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   芯片   | 卡数 |  Acc@1  | FPS  | Max epochs |
|:------:|----|:------:|:----:|:----------:|
|  竞品A   | 1p |   -   | 731.2  |   1    |
|  竞品A   | 8p | 74.23 | 5314.9 |  100   |
| Atlas 800T A2 | 1p |   -   | 694.9  |   1    |
| Atlas 800T A2 | 8p | 74.29 | 5150.5 |  100   |


# 版本说明

## 变更
2024.01.30：更新readme，重新发布。
2024.02.29：补充数据集预处理操作说明。

## FAQ

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
