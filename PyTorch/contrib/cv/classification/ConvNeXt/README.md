# ConvNext_for_PyTorch

This implements training ConvNext of  on the ImageNet dataset, mainly modified from https://github.com/facebookresearch/ConvNeXt.git 

## ConvNext_for_PyTorch Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 



## Requirements 
- pip install -r requirements.txt
- pip install torch==1.8.1+ascend.rc2.20220505;torchvision==0.9.1;torch-npu 1.8.1rc2.post20220505;
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
## timm 
将timm_need目录下的文件替换到timm的安装目录下
```bash

cd ../ConvNeXt
/bin/cp -f timm_need/mixup.py ../timm/data/mixup.py
/bin/cp -f timm_need/model_ema.py ../timm/utils/model_ema.py

```

## 软件包

该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
2. 查看[软件版本配套表](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitee.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**

当前模型支持的历史版本软件如下所示。
- 910版本
- CANN toolkit_5.1.RC1
- torch 1.8.1+ascend.rc2.20220505
- 固件驱动 22.0.0

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

#  eval 
bash test/train_eval_8p.sh --data_path=real_data_path

# finetuning
bash test/train_finetune_1p.sh --data_path=real_data_path


```

## ConvNext_for_PyTorch training result 

|  Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type |
| :-----: | :------: | :------: | :----: | :------: |
|    -    |    115.10      |    1     |  300   |    O1    |
| 82.049 | 259.85 |    8     |  300   |    O1    |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md




