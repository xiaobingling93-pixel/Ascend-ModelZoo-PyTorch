---
license: mit
pipeline_tag: multi-modal
hardwares:
  - NPU
frameworks:
  - PyTorch
library_name: openmind
language:
  - en
---
## 版本配套
| 组件 | 版本 |
| - | - |
| MindIE | 1.0.0 |
| CANN | 8.0.0 |
| PTA | 6.0.0 |
| MindStudio | 7.0.0 |
| HDK | 24.1.0 |
## 一、准备运行环境

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持：
Atlas 800I A2推理设备：支持的卡数最小为1
Atlas 300I Duo推理卡：支持的卡数最小为1
Atlas 300 V:支持的卡数最小为1
- [Atlas 800I A2/Atlas 300I Duo/Atlas 300 V](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
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

### 1.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.4 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```
## 二、执行推理
### 2.1 下载权重
### Huggingface
| Model                 | Sequence Length | Download                                                                    |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| Janus-Pro-1B | 4096            | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-1B) |
| Janus-Pro-7B | 4096        | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-7B) |
### 2.2 下载模型依赖
```shell
pip install -e .
```
### 2.3 执行推理脚本

#### 2.3.1 多模态理解
```python
python inference.py --path './deepseek-ai/Janus-Pro' --device_id 0 --type bf16
```

#### 2.3.2 多模态生成
```python
python generation_inference.py --path './deepseek-ai/Janus-Pro' --device_id 0 --type bf16
```

#### 2.3.3 命令行参数说明
```python
--device_id 指定npu运行设备
--type 可指定bf16或fp16，Atlas 300I Duo/Atlas 300I Pro/Atlas 300 V设备只支持fp16
--path 指定模型路径
```

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。