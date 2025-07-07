## 一、准备运行环境
### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800T A2(64G)：支持的卡数最小为8
Atlas 800I A2(64G)：支持的卡数最小为8
- [Atlas 800T A2(64G)/Atlas 800I A2(64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
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

### 1.5 安装gcc、g++
```shell
# 若环境镜像中没有gcc、g++，请用户自行安装
yum install gcc
yum install g++

# 导入头文件路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
```

## 二、下载权重

### 2.1 权重及配置文件说明
stepvideo-ti2v权重链接:

```shell
# modelers
https://modelers.cn/models/StepFun/stepvideo-ti2v
```
```shell
# huggingface
https://huggingface.co/stepfun-ai/stepvideo-ti2v
```

## 三、StepVideo-TI2V使用
当前支持的分辨率：544x992、768x768
### 3.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/Step-Video-TI2V.git
```
### 3.2 安装依赖
```shell
cd StepVideo-TI2V
pip install -e .
```
安装xfuser
```shell
git clone -b 0.4.2 https://github.com/xdit-project/xDiT.git
bash patch.sh
```

### 3.3 8卡性能测试
#### 3.3.1 等价优化
执行命令：
```shell
# 使用上一步下载的权重
export model_dir='./stepvideo-ti2v/'
export ALGO=1
export TASK_QUEUE_ENALBLE=2
export CPU_AFFINITY_CONF=2
export HCCL_RDMA_TC=144
export LCCL_DETERMINISTIC=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="{path_to_atbops}:$PYTHONPATH" # {path_to_atbops}指atb_ops的上层目录路径

# 执行推理
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 run_parallel.py \
--model_dir ${model_dir} \
--height 544 \
--width 992 \
--num_frames 102 \
--infer_steps 30  \
--ulysses_degree 1 \
--tensor_parallel_degree 8 \
--prompt="带翅膀的小老鼠先用爪子挠了挠脑袋，随后扑扇着翅膀飞了起来。"  \
--first_image_path './benchmark/Step-Video-TI2V-Eval/ti2v_eval_real/S/061.png' \ # 修改图片路径
--save_path './results'
```

参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- model_dir: 配置文件及权重路径。
- height: 生成视频的高
- weight: 生成视频的宽
- num_frames: 视频帧数
- infer_steps: 推理步数
- ulysses_degree: ulysses并行度
- tensor_parallel_degree: tp并行度
- prompt: 文本提示词
- first_image_path: 用于生成视频的图片路径
- save_path: 生成的视频的保存路径

#### 3.3.2 算法优化
使用DiTcache
执行命令：
```shell
# 使用上一步下载的权重
export model_dir='./stepvideo-ti2v/'
export ALGO=1
export TASK_QUEUE_ENALBLE=2
export CPU_AFFINITY_CONF=2
export HCCL_RDMA_TC=144
export LCCL_DETERMINISTIC=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="{path_to_atbops}:$PYTHONPATH" # {path_to_atbops}指atb_ops的上层目录路径

# 执行推理
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 run_parallel.py \
--model_dir ${model_dir} \
--height 544 \
--width 992 \
--num_frames 102 \
--infer_steps 30  \
--ulysses_degree 1 \
--tensor_parallel_degree 8 \
--prompt="带翅膀的小老鼠先用爪子挠了挠脑袋，随后扑扇着翅膀飞了起来。"  \
--first_image_path './benchmark/Step-Video-TI2V-Eval/ti2v_eval_real/S/061.png' \ # 修改图片路径
--save_path './results' \
--use_dit_cache
```

参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- model_dir: 配置文件及权重路径。
- height: 生成视频的高
- weight: 生成视频的宽
- num_frames: 视频帧数
- infer_steps: 推理步数
- ulysses_degree: ulysses并行度
- tensor_parallel_degree: tp并行度
- prompt: 文本提示词
- first_image_path: 用于生成视频的图片路径
- save_path: 生成的视频的保存路径
- use_dit_cache: 使用DiTcache策略

## 四、模型推理性能结果参考
### StepVideo-TI2V
NPU端到端性能和GPU进行了对比，平均每卡的吞吐达到1.5x GPU A800。性能测试如下：
| 硬件形态  | 分辨率 | GPU数（NPU数） | 迭代次数 | 端到端耗时 | 单卡吞吐（fps/p） |
| :------: | :------: | :------: |:----:| :------: | :------: |
| A800 | 544px × 992px × 102f | 5 | 30  | 491s | 0.041547 |
| Atlas 800T A2(64G)| 544px × 992px × 102f | 8 | 30  | 204.17s | 0.062445 |

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。