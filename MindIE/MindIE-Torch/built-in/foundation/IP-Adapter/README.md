# 模型推理指导  

## 一、模型简介

在Stable Diffusion研究中，如何有效地将文本提示和图像提示整合到预训练的文生图模型中一直一个挑战。IPAdapter通过引入一个轻量级的适配器模块创新地解决了这个问题，参考实现请查看[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)。

## 二、环境准备

  **表 1**  版本配套表

  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
  | torch| 2.1.0  | -                                                            |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- 支持卡数：Atlas 800I A2支持的卡数为1；Atlas 300I Duo支持的卡数为1
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)
**注意**：该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

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

### 2.4 代码下载
1. 本仓库代码下载
```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

2. 获取源码，并将其中部分文件移到IP-Adapter工程下
```bash
git clone https://github.com/tencent-ailab/IP-Adapter
mv attention_processor.patch clip.patch export_ts.py inference.py stable_diffusion_patch.py stable_diffusion_pipeline.py requirements.txt ./IP-Adapter
```

### 2.5 安装所需依赖
按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
```shell
cd IP-Adapter
pip3 install -r requirements.txt
```

## 三、模型权重

下载权重，放在任意路径，以避免执行后面步骤时可能会出现下载失败。

[runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)

git clone https://huggingface.co/h94/IP-Adapter IP-Adapter-weights

## 四、模型推理

### 4.1 代码修改
执行命令：
```bash
python3 stable_diffusion_patch.py
```

### 4.2 模型转换
使用Pytorch导出pt模型，再使用MindIE推理引擎转换为适配昇腾的模型。

1. 设置权重路径：
   设置模型名称或路径
   ```bash
   # v1.5
   base_model_path="runwayml/stable-diffusion-v1-5"
   
   # vae
   vae_model_path="stabilityai/sd-vae-ft-mse"
   
   # image_encoder
   image_encoder_path="IP-Adapter-weights/models/image_encoder"
   
   # ip_ckpt
   ip_ckpt="IP-Adapter-weights/models/ip-adapter_sd15.bin"
   ```

2. 执行命令：
   ```bash
   # 导出pt模型
   python3 export_ts.py \
   --base_model_path ${base_model_path} \
   --vae_model_path ${vae_model_path} \
   --image_encoder_path ${image_encoder_path} \
   --batch_size 1 \
   --output_dir ./models \
   --device 0 \
   --soc Duo
   ```
   
   参数说明：
   - --base_model_path：SD的模型名称或本地模型目录的路径
   - --vae_model_path：VAE的模型名称或本地模型目录的路径
   - --image_encoder_path：image_encoder的模型名称或本地模型目录的路径
   - --output_dir: 导出的模型输出目录
   - batch_size：目前只支持batch为1
   - --device：使用的NPU芯片，默认是0
   - soc：soc_version。默认为Duo，可支持A2
      
### 4.3 性能测试
1. 开启cpu高性能模式
   ```bash
   echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   sysctl -w vm.swappiness=0
   sysctl -w kernel.numa_balancing=0
   ```

2. 安装绑核工具
   ```bash
   apt-get update
   apt-get install numactl
   ```
   查询卡的NUMA node
   ```shell
   lspci -vs bus-id
   ```
   bus-id可通过npu-smi info获得，查询到NUMA node，在推理命令前加上对应的数字

   可通过lscpu获得NUMA node对应的CPU核数
   ```shell
   NUMA node0: 0-23
   NUMA node1: 24-47
   NUMA node2: 48-71
   NUMA node3: 72-95
   ```
   当前查到NUMA node是0，对应0-23，推荐绑定其中单核以获得更好的性能。

3. 执行推理脚本
   ```bash
   numactl -C 0-23 python3 inference.py \
   --base_model_path ${base_model_path} \
   --vae_model_path ${vae_model_path} \
   --image_encoder_path ${image_encoder_path} \
   --ip_ckpt ${ip_ckpt} \
   --output_dir ./models \
   --device 0 \
   --image_path ./assets/images/woman.png \
   --save_image_path ./test.png \
   --prompt "A girl"
   ```
   
   参数说明：
   - --base_model_path：SD的模型名称或本地模型目录的路径
   - --vae_model_path：VAE的模型名称或本地模型目录的路径
   - --image_encoder_path：image_encoder的模型名称或本地模型目录的路径
   - --ip_ckpt：ipadpter的模型名称或本地模型目录的路径
   - --output_dir: 导出的模型输出目录
   - --device：使用的NPU芯片，默认是0
   - --image_path: 输入的图片路径
   - --save_image_path：输出的图片路径
   - --prompt：文本提示词


## 五、推理结果参考
### IP-Adapter性能数据

| 硬件形态 | batch size| 迭代次数 | 性能 |
| :------: |:----:|:----:|:----:|
| Atlas 300I Duo |  1  |  50  | 4.09s |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。