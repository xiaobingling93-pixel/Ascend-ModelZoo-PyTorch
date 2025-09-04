# pytorch_AdvancedEast
pytorch实现AdvancedEast+mobilenetv3

# 参考https://github.com/huoyijie/AdvancedEAST 
# training
## 用户自行准备数据集
### 1.modify config params in cfg.py, see default values.
### 2.python preprocess.py, resize image to 256256,384384,512512,640640,736*736, and train respectively could speed up training process.
### 3.python label.py
### 4.python train.py
### 5.python predict.py
图片:
![demo](https://github.com/corleonechensiyu/pytorch_AdvancedEast/blob/master/012.png_predict.jpg)

