# Janus-Pro-1B-OrangePi
## 版本配套
| 组件 | 版本 |
| - | - |
| MindIE | 1.0.0 |
| CANN | 8.0.0 |
| PTA | 6.0.0 |
| MindStudio | 7.0.0 |
| HDK | 24.1.0 |
| python | 3.10 |
## 一、准备运行环境

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
OrangePi AiPro 20T/24G
- 环境准备
[环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)
- 安装包下载
[PyTorch下载链接](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann)
请下载```Ascend Extension for PyTorch - 6.0.0.beta1-PyTorch2.1.0```中的``` torch_npu-2.1.0.post10-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl```
- [CANN-toolkit下载链接](https://mindie.obs.cn-north-4.myhuaweicloud.com/xiangchengpai_20250211/Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run)
- [CANN-kernels下载链接](https://mindie.obs.cn-north-4.myhuaweicloud.com/xiangchengpai_20250211/Ascend-cann-kernels-310b_8.1.RC1_linux-aarch64.run)
- 代码下载
```git clone https://modelers.cn/MindIE/Janus-Pro-1B-OrangePi.git```

### 1.2 CANN安装
```shell
# 增加软件包可执行权限
chmod +x ./Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run
chmod +x ./Ascend-cann-kernels-310b_8.1.RC1_linux-aarch64.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --check
./Ascend-cann-kernels-310b_8.1.RC1_linux-aarch64.run --check
# 安装
./Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run  --install --force
./Ascend-cann-kernels-310b_8.1.RC1_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 1.3 Torch_npu和相关依赖安装
```shell
pip install -r requirements.txt
pip install torch_npu-2.1.0.post10-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```
替换transformers中的utils.py
```
transformers_path=$(pip3 show transformers|grep Location|awk '{print $2}')
# 修改${transformers_path}/transformers/generation/utils.py中的函数
# 将第2487行修改为
model_kwargs["cache_position"] = torch.arange(cur_len, device="cpu").npu()
```
替换transformers中的modeling_llama.py
```
transformers_path=$(pip3 show transformers|grep Location|awk '{print $2}')
# 修改${transformers_path}/transformers/models/llama/modeling_llama.py中的函数
# 将第141行修改为
freqs = (inv_freq_expanded.to(torch.float16) @ position_ids_expanded.to(torch.float16)).transpose(1, 2)
# 将第985~987行修改为
cache_position = torch.arange(
    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device="cpu"
).npu()
# 将第1096行修改为
causal_mask *= torch.arange(target_length, device="cpu").npu() > cache_position.reshape(-1, 1)
```
替换timm中的mlp.py
```
timm_path=$(pip3 show timm|grep Location|awk '{print $2}')
# 修改${timm_path}/mlp/layers/mlp.py中的函数
# 将第45行修改为
x = self.act(x.cpu()).npu()
```
## 二、执行推理
### 2.1 下载权重
- [Janus-Pro-1B](https://modelers.cn/models/State_Cloud/Janus-Pro-1B)
### 2.2 下载模型依赖
```shell
pip install -e .
```
### 2.3 执行推理脚本

#### 2.3.1 多模态理解
```python
python inference.py --path './deepseek-ai/Janus-Pro' --device_id 0 --type fp16
```

#### 2.3.2 多模态生成
```python
python generation_inference.py --path './deepseek-ai/Janus-Pro' --device_id 0 --type fp16
```

#### 2.3.3 命令行参数说明
```python
--device_id 指定npu运行设备
--type 运行的数据类型，设备只支持fp16
--path 指定模型路径
```