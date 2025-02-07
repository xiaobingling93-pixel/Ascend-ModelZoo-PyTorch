# EfficientNetV2-B0_for_Pytorch

# 目录
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [预训练数据集](#预训练数据集)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [训练结果](#训练结果)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)


# 简介


## 模型介绍

*EfficientNetV2*是*Efficient*的改进版，accuracy达到了发布时的SOTA水平，而且训练速度更快参数来更少。相对EfficientNetV1系列只关注准确率，参数量以及FLOPs，V2版本更加关注模型的实际训练速度。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/huggingface/pytorch-image-models.git
  commit_id=6e6f3686a7e06bcba37bbd3b7c755f04a516a1e7
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
# 准备训练环境

该模型为随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），您可以根据下面提供的安装指导选择匹配的CANN等软件下载使用。

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

- 安装依赖

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  
  ```
  # pytorch 2.1请使用requirements_2_1.txt
  pip install -r requirements_2_1.txt

  ```

# 准备数据集

## 预训练数据集
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

# 快速开始

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
     bash ./test/train_1p.sh --data_path=指向imagenet数据集 # 单卡精度
     
     bash ./test/train_1p.sh --data_path=指向imagenet数据集 --performance=1  # 单卡性能
     ```
   
- 单机8卡训练

     启动8卡训练。
   
     ```
     bash ./test/train_8p.sh --data_path=指向imagenet数据集  # 8卡精度

     bash ./test/train_8p.sh --data_path=指向imagenet数据集 --performance=1 # 8卡性能 
     ```

   --data_path参数填写数据集路径，需写到imagenet的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                              //数据集路径
   --performance                            //--performance=1开启性能测试，默认不开启
   --batch_size                             //单卡batch_size，默认为256
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


## 训练结果

**表 2**  训练结果展示表

|  芯片      | 卡数 | Acc@1 |   FPS   | Epochs |
|:--------:|----|:-----:|:-------:|:------:|
|   竞品A    | 1p |   -   | 1047.13 |   1    |
|   竞品A    | 8p | 72.91 | 5707.58 |  100   |
| Atlas 800T A2 | 1p |   -   | 867.83  |   1    |
| Atlas 800T A2 | 8p | 72.68 | 6274.66 |  100   |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明
2024.03.25：首次发布。

## FAQ
暂无。



