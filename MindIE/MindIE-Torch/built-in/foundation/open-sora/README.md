# 模型推理指导  

## 一、模型简介

OpenSora是一种文本到视频的扩散模型，能够在给定文本输入的情况下生成相符的视频。OpenSora采用动态掩码策略等技术细节复现Sora，并已实现可变长宽比、可变分辨率和可变时长等功能。

本模型输入输出数据：
输入一个prompt，输出一个2s长的视频

## 二、环境准备
  **表 1**  版本配套表

  | 配套                                 | 版本    | 环境准备指导 |
  | ------------------------------------ | ------- | ------------ |
  | Python                               | 3.10.13 | -            |
  | PyTorch                              | 2.1.0   | -            |
  | 硬件：Atlas 300I Duo ，Atlas 800I A2 | \       | \            |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- 支持卡数：Atlas 800I A2支持的卡数为1；Atlas 300I Duo支持的卡数为1
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

### 2.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 2.4 下载本仓库
```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

### 2.5 安装所需依赖
按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
```shell
pip install -r requirements.txt
```

## 三、模型权重

ST-DIT权重文件下载链接如下，按需下载：

[ST-DIT-256x256下载链接](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth)

[ST-DIT-512x512下载链接](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)

vae权重文件下载链接如下，按需下载：

```bash
# ema
git clone https://huggingface.co/stabilityai/sd-vae-ft-ema
```

encoder权重文件

```bash
https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main
```

## 四、模型推理

### 4.1 模型转换
使用Pytorch导出pt模型，再使用MindIE推理引擎转换为适配昇腾的模型。
```bash
python3 export_model.py \
--output_dir ./models \
--encoder_model_path ./DeepFloyd--t5-v1_1-xxl \
--dit_model_path ./OpenSora-v1-HQ-16x512x512.pth \
--vae_model_path ./sd-vae-ft-ema \
--resolution 16x512x512 \
--device_id 0
```

参数说明：

- --encoder_model_path：encoder的权重路径
- --dit_model_path：dit的权重路径
- --vae_model_path：vae的权重路径
- --resolution：分辨率。支持256和512
- --device_id：NPU芯片
- --output_dir：pt模型输出目录

### 4.2 性能测试
1. 开启cpu高性能模式：
   ```bash
   echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   sysctl -w vm.swappiness=0
   sysctl -w kernel.numa_balancing=0
   ```

2. 执行命令：
   ```bash
   python inference.py \
   ./configs/opensora/inference/16x256x256.py \
   --ckpt-path ./OpenSora-v1-HQ-16x512x512.pth \
   --prompt-path ./assets/texts/t2v_samples.txt \
   --use_mindie 1 \
   --device_id 0
   ```
   
   参数说明：
   
   - --ckpt-path：STDIT的权重路径
   - --prompt-path：prompt数据集的路径
   - --use_mindie：是否使用MindIE推理。1代表是，0代表否
   - --device_id：使用哪张卡

执行完成后会在当前路径生成sample.mp4。


## 五、推理结果参考
### OpenSora 性能数据

| 分辨率 | 硬件形态 | 性能 |
| ------ | -------- | -------- |
| 512    | Atlas 800I A2(8*32G) | 110.8s   |
| 256    | Atlas 300I Duo  | 22.2s    |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。