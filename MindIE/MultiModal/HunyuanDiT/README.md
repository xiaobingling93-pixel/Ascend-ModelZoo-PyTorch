# 模型推理指导  

## 一、模型简介

HunyuanDiT是一种文本到图像的扩散模型，能够在给定文本输入的情况下生成相符的图像。

  **表 1**  本模型当前支持的分辨率

  | 比例（H:W） | 1:1 | 1:1 | 4:3 | 4:3 | 4:3 | 3:4 | 3:4 | 3:4 | 5:3 | 3:5 |
  | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
  | 分辨率 | 1024×1024 | 1280×1280 | 1024×768 | 1152×864 | 1280×960 | 768×1024 | 864×1152 | 960×1280 | 1280×768 | 768×1280 |

本模型使用的优化手段如下：
- 等价优化：FA、RoPE、Linear
- 算法优化：FA、RoPE、Linear、Cache

## 二、环境准备

  **表 2**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 800T A2](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26)
- 支持卡数：支持的卡数为1
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

### 2.4 Torch_npu安装
安装pytorch框架 版本2.1.0
[安装包下载](https://download.pytorch.org/whl/cpu/torch/)

使用pip安装
```shell
# {version}表示软件版本号，{arch}表示CPU架构。
pip install torch-${version}-cp310-cp310-linux_${arch}.whl
```
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

### 2.5 下载本仓库
```shell
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

### 2.6 安装所需依赖
```shell
pip install -r requirements.txt

# 若要使用hpsv2验证精度，则还需要按照以下步骤安装hpsv2
git clone https://github.com/tgxs002/HPSv2.git
pip install -e HPSv2
```

## 三、模型权重

### 3.1 权重下载
```shell
https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2/tree/main/t2i
```

### 3.2 配置文件说明
1. 在t2i/model路径下，新增HunyuanDiT模型权重的配置文件，命名为config.json
```shell
{
  "_class_name": "HunyuanDiT2DModel",
  "_mindiesd_version": "2.0.RC1",
  "input_size": [
    null,
    null
  ],
  "patch_size": 2,
  "in_channels": 4,
  "hidden_size": 1408,
  "depth": 40,
  "num_heads": 16,
  "mlp_ratio": 4.3637,
  "text_states_dim": 1024,
  "text_states_dim_t5": 2048,
  "text_len": 77,
  "text_len_t5": 256,
  "size_cond": null,
  "use_style_cond": false
}
```
2. 各模型的配置文件、权重文件的路径层级样例如下所示。
```commandline
|----hunyuan_dit
|    |---- ckpts
|    |    |---- t2i
|    |    |    |---- clip_text_encoder
|    |    |    |---- model
|    |    |    |    |---- config.json
|    |    |    |    |---- 模型权重
|    |    |    |---- mt5
|    |    |    |---- sdxl-vae-fp16-fix
|    |    |    |---- tokenizer
```

## 四、模型推理

### 4.1 单卡推理适配性能测试
1. 设置权重路径：
```shell
path="ckpts/t2i"
```
修改权重和prompts文件夹的权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
chmod -R 640 prompts/
```

2. 执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --prompt "渔舟唱晚" \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- prompt：用于图像生成的文字描述提示。
- input_size：生成的图像尺寸，宽高要求是8的倍数。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在`results`目录下生成一张推理图像。

### 4.2 单卡等价优化推理性能测试
1. 设置权重路径
```shell
path="ckpts/hydit"
```
修改权重和prompts文件夹的权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
chmod -R 640 prompts/
```

2. 执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --test_acc \
       --prompt_file "prompts/example_prompts.txt" \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启prompt_file列表中的图像生成，用于性能/精度测试。
- prompt_file：用于图像生成的文字描述提示的列表文件路径。
- input_size：生成的图像尺寸，宽高要求是8的倍数。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在`results`目录下生成推理图像，图像生成顺序与prompt顺序保持一致，并在终端显示推理时间。

### 4.3 单卡算法优化推理性能测试
1. 设置权重路径
```shell
path="ckpts/hydit"
```
修改权重和prompts文件夹的权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
chmod -R 640 prompts/
```
2. 执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --test_acc \
       --prompt_file "prompts/example_prompts.txt" \
       --use_attentioncache \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启prompt_file列表中的图像生成，用于性能/精度测试。
- prompt_file：用于图像生成的文字描述提示的列表文件路径。
- use_attentioncache：使用 --use_attentioncache 开启算法策略优化的测试。
- input_size：生成的图像尺寸，宽高要求是8的倍数。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在`results`目录下生成推理图像，图像生成顺序与prompt顺序保持一致，并在终端显示推理时间。

### 4.4 单卡多batch推理性能测试
1. 设置权重路径
```shell
path="ckpts/hydit"
```
修改权重和prompts文件夹的权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
chmod -R 640 prompts/
```
2. 执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --test_acc \
       --prompt_file "prompts/example_prompts.txt" \
       --use_attentioncache \
       --input_size 1024 1024 \
       --batch_size 2 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启prompt_file列表中的图像生成，用于性能/精度测试。
- prompt_file：用于图像生成的文字描述提示的列表文件路径。
- use_attentioncache：使用 --use_attentioncache 开启算法策略优化的测试。
- input_size：生成的图像尺寸，宽高要求是8的倍数。
- batch_size：每个prompt生成的图像数量，根据设备显存，batch_size最大设置为2。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在`results`目录下生成推理图像，图像生成顺序与prompt顺序保持一致，并在终端显示推理时间。

### 4.5 精度测试
由于生成的图像存在随机性，提供两种精度验证方法：
- CLIP-score（文图匹配度量）：评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。使用Parti数据集进行验证
- HPSv2（图片美学度量）：评估生成图片的人类偏好评分，分数的取值范围为[0, 1]，越高越好。使用HPSv2数据集进行验证

**注意**：由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

1. 下载Parti数据集和hpsv2数据集
```bash
# 下载Parti数据集
wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
```
hpsv2数据集下载链接：https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_benchmark_prompts.json

建议将`PartiPrompts.tsv`和`hpsv2_benchmark_prompts.json`文件放到`prompts/`路径下。

2. 下载模型权重
```bash
# Clip Score和HPSv2均需要使用的权重
# 安装git-lfs
apt install git-lfs
git lfs install

# Clip Score权重
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

# HPSv2权重
wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt --no-check-certificate
```
也可手动下载[权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)
将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下，手动下载[HPSv2权重](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)放到当前路径

3. 使用推理脚本读取Parti数据集，生成图片
设置权重路径
```shell
path="ckpts/hydit"
```
修改权重和prompts文件夹的权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
chmod -R 640 prompts/
```
执行命令：
```shell
# 使用算法优化
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --test_acc \
       --prompt_file "prompts/PartiPrompts.tsv" \
       --prompt_file_type parti \
       --max_num_prompts 0 \
       --info_file_save_path ./image_info_parti.json \
       --save_result_path ./results_parti \
       --use_attentioncache \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启prompt_file列表中的图像生成，用于性能/精度测试。
- prompt_file：用于图像生成的文字描述提示的列表文件路径。
- prompt_file_type：prompt文件类型，用于指定读取方式，可选范围：plain，parti，hpsv2。默认值为plain。
- max_num_prompts：限制prompt数量为前X个，0表示不限制。
- info_file_save_path：生成图像信息的json文件路径。
- save_result_path：生成图像的存放目录。
- use_attentioncache：使用 --use_attentioncache 开启算法策略优化的测试。
- input_size：生成的图像尺寸，宽高要求是8的倍数。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在`./results_parti`目录下生成推理图像。在当前目录下生成一个`image_info_parti.json`文件，记录着图像和prompt的对应关系，并在终端显示推理时间。

4. 使用推理脚本读取hpsv2数据集，生成图片
设置权重路径
```shell
path="ckpts/hydit"
```
修改权重和prompts文件夹的权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
chmod -R 640 prompts/
```
执行命令：
```shell
# 使用算法优化
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --test_acc \
       --prompt_file "prompts/hpsv2_benchmark_prompts.json" \
       --prompt_file_type hpsv2 \
       --max_num_prompts 0 \
       --info_file_save_path ./image_info_hpsv2.json \
       --save_result_path ./results_hpsv2 \
       --use_attentioncache \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启prompt_file列表中的图像生成，用于性能/精度测试。
- prompt_file：用于图像生成的文字描述提示的列表文件路径。
- prompt_file_type：prompt文件类型，用于指定读取方式，可选范围：plain，parti，hpsv2。默认值为plain。
- max_num_prompts：限制prompt数量为前X个，0表示不限制。
- info_file_save_path：生成图像信息的json文件路径。
- save_result_path：生成图像的存放目录。
- use_attentioncache：使用 --use_attentioncache 开启算法策略优化的测试。
- input_size：生成的图像尺寸，宽高要求是8的倍数。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在`./results_hpsv2`目录下生成推理图像。在当前目录下生成一个`image_info_hpsv2.json`文件，记录着图像和prompt的对应关系，并在终端显示推理时间。

5. 计算精度指标
   1. CLIP-score
       ```bash
       python clip_score.py \
              --device=cpu \
              --image_info="image_info_parti.json" \
              --model_name="ViT-H-14" \
              --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
       ```
       参数说明：
       - device: 推理设备，默认为"cpu"，如果是cuda设备可设置为"cuda"。
       - image_info: 上一步生成的`image_info_parti.json`文件。
       - model_name: Clip模型名称。
       - model_weights_path: Clip模型权重文件路径。

       clip_score.py脚本可参考[SDXL](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/clip_score.py)，执行完成后会在屏幕打印出精度计算结果。

   2. HPSv2
      ```bash
      python hpsv2_score.py \
            --image_info="image_info_hpsv2.json" \
            --HPSv2_checkpoint="./HPS_v2_compressed.pt" \
            --clip_checkpoint="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
      ```

      参数说明：
      - image_info: 上一步生成的`image_info_hpsv2.json`文件。
      - HPSv2_checkpoint: HPSv2模型权重文件路径。
      - clip_checkpointh: Clip模型权重文件路径。

      hpsv2_score.py脚本可参考[SDXL](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_score.py)，执行完成后会在屏幕打印出精度计算结果。

## 五、推理结果参考
### HunyuanDiT性能 & 精度数据

| 硬件形态 | cpu规格 | 分辨率 | batch size | 迭代次数 | 优化方式 | 平均耗时 | CLIP-score | HPSv2-score |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Atlas 800I A2(8*32G) | 64核(arm) | 1024×1024 | 1 | 100 | 等价优化 | 42.413s | 0.339 | 0.2843779 |
| Atlas 800I A2(8*32G) | 64核(arm) | 1024×1024 | 1 | 100 | 算法优化 | 25.373s | 0.339 | 0.2825251 |

注意：性能测试需要独占npu和cpu


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。