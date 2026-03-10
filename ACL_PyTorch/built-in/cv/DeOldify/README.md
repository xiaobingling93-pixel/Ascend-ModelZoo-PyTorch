# DeOldify(aclgraph)-推理指导

- [DeOldify(aclgraph)-推理指导](#DeOldify(aclgraph)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [环境配置](#环境配置)
  - [获取权重](#获取权重)
  - [获取测试数据](#获取测试数据)
  - [模型推理](#模型推理)
  - [结果展示](#结果展示)


# 概述
DeOldify 是一种著名的自动上色开源算法。该模型采用 ResNet 作为编码器构建了具有 UNet 结构的网络，并提出了几个不同的训练版本，在效果、效率、鲁棒性等方面具有良好的综合性能。

# 推理环境准备

| 配套 | 版本 | 环境指导准备 |
|--|--|--|
| 固件与驱动 | 25.5.1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies/pies_00001.html) |
| CANN | 8.5.0 | - |
| torch | 2.9.0 | - |
| torch-npu | 2.9.0 | - |
| torchvision | 0.24.0 | - |
| 说明：该模型暂不适用于Atlas 300I系列产品 | - | - |

# 快速上手
## 环境配置
1. 安装依赖
```
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/DeOldify
git clone https://github.com/jantic/DeOldify.git(或者git clone https://gitcode.com/gh_mirrors/de/DeOldify.git)
cd DeOldify
patch -p1 < ../diff.patch
apt-get update
apt-get install -y libgl1 libglib2.0-0 ffmpeg
pip install -r requirements.txt
cd ..
cp infer.py DeOldify
cd DeOldify
pip uninstall triton-ascend
```

## 获取权重
下载DeOldify模型权重，放置于本地目录`models`中

[artistic](https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth)  
[stable](https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=0)  
[video](https://data.deepai.org/deoldify/ColorizeVideo_gen.pth)

## 获取测试数据
下载测试图片，放置于本地目录`test_images`中
```
cd test_images
wget https://media.githubusercontent.com/media/dana-kelley/DeOldify/refs/heads/master/test_images/1860Girls.jpg
cd ..
```

如需做视频上色，则下载测试视频，放置于本地目录`video/source`中（如果没有就自行创建目录）

## 模型推理
### 设置环境变量
```
# 开启虚拟内存
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 队列优化特性
export TASK_QUEUE_ENABLE=1
# CPU绑核
export CPU_AFFINITY_CONF=1 
```
### 推理命令
```
ASCEND_RT_VISIBLE_DEVICES=X python infer.py \
--image_artistic \
--image \
--video \
--image_path "test_images/XXX.jpg" \
--image_url "https://XXX.jpg" \
--video_path "XX.mp4" \
--video_url "httpsL//XX.mp4" \
--image_render_factor 35 \
--video_render_factor 21 \
--watermarked \
--warm_num 2 \
--infer_num 3
```
参数说明  
ASCEND_RT_VISIBLE_DEVICES：推理时使用的卡，仅支持单卡，仅能输入0，1，2这样的数字。  
image_artistic：图片上色时是否使用artistic模型。  
image：进行图片上色。  
video：进行视频上色。  
image_path：本地带上色图片路径，与image_url同时使用时不生效。  
image_url：带上色图片网址，优先级高于image_path，两个参数共存时image_path不生效。  
video_path：本地带上色视频路径，与video_url同时使用时不生效。  
video_url：带上色图片网址，优先级高于video_path，两个参数共存时video_path不生效。  
image_render_factor：图片渲染因子，数值越高，着色品质越好，但是处理时间越长，默认值35。  
video_render_factor：视频渲染因子，数值越高，着色品质越好，但是处理时间越长，默认值21。  
watermarked：添加水印。  
warm_num：预热次数，预热可提升模型推理性能。  
infer_num：每个输入推理次数。  

比如推理图片
```
ASCEND_RT_VISIBLE_DEVICES=0 python infer.py \
--image_artistic \
--image \
--image_path "test_images/1860Girls.jpg" \
--image_render_factor 35
```
第一次推理时会自动下载所需的辅助模型
![模型下载.png](https://raw.gitcode.com/user-images/assets/9336645/ea0c21e9-ecd5-4674-8175-bbf0c4aff07c/模型下载.png '模型下载.png')
## 结果展示
本次测试使用Atlas 800I A2 64G单卡，64核CPU，在quay.io/ascend/vllm-ascend:v0.15.0rc1上推理结果如下：  
### 推理性能

| 图片尺寸 | 封装格式 | render_factor | 并发数 | 推理时长 |
|--|--|--|--|--|
| 720p | png | 35 | 1 | 0.6s |
| 1080p | jpg | 35 | 1 | 0.3s |

| 视频规格 | 封装格式 | render_factor | 并发数 | 推理时长 |
|--|--|--|--|--|
|20s 720p@30fps | mp4 | 21 | 1 | 105s |

### 原始图片
![1860Girls.jpg](https://raw.gitcode.com/user-images/assets/9336645/f8c74bd0-ab52-4ec3-b54d-5e06dc49136b/1860Girls.jpg '1860Girls.jpg')
### 图片推理生成结果
![1860Girls-new.jpg](https://raw.gitcode.com/user-images/assets/9336645/ffd513c4-f5f0-4a19-99e4-1207d34175b5/1860Girls-new.jpg '1860Girls-new.jpg')