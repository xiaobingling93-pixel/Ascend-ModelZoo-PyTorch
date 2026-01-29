# openPangu-Embedded-1B-OrangePi
## 简介
openPangu-Embedded-1B 是基于昇腾 NPU 从零训练的高效语言模型，参数量为 1B（不含词表Embedding），模型结构采用 26 层 Dense 架构，训练了约 10T tokens。通过昇腾 Atlas 200I A2可用的模型架构设计、数据和训练策略优化，openPangu-Embedded-1B 在保持端侧运行的要求下达到了较高的精度。

## 约束条件
* 在OrangePi AIpro(20T)上部署openPangu-Embedded-1B模型
* 需要修改权重目录下的config.json文件，"torch_dtype"字段改为"float16", "max_position_embedding"字段改为4096, 删除“rope_scaling”字段
* 由于此硬件为单卡，仅支持TP=1

## 权重

**权重下载**

- [openPangu-Embedded-1B](https://ai.gitcode.com/ascend-tribe/openpangu-embedded-1b-model/tree/main)

## 环境准备

### 1. 虚拟环境

- 创建虚拟环境：
    ```shell
    conda create -n env_name python=3.10
    ```
  
- 获取设备信息
  - 使用`uname -a`指令查看服务器是x86还是aarch架构
  - 使用以下指令查看abi是0还是1
    ```shell
    python -c "import torch; print(torch.compiled_with_cxx11_abi())"
    ```
    若输出结果为True表示abi1，False表示abi0
    
### 2. 资源下载

请前往[昇腾社区/社区版资源下载](https://www.hiascend.com/developer/download/community/result?module=ie+pt+cann)下载适配板卡的MindIE、CANN和PTA组件，各版本配套表如下：
| 组件 | 版本 |
| - | - |
| MindIE | 2.1.RC1 |
| CANN | 8.2.RC1 |
| PTA | 7.1.0 |

- CANN下载内容：toolkit(工具包)；kernels（算子包）；nnal(加速库)
- PTA下载内容：torch_npu

### 3. 安装CANN

- 安装顺序：先安装toolkit 再安装kernel 最后安装nnal

#### 3.1 安装toolkit

- 检查包

| cpu     | 包名（其中`${version}`为实际版本）                 |
| ------- | ------------------------------------------------ |
| aarch64 | Ascend-cann-toolkit_${version}_linux-aarch64.run |

- 安装
  ```bash
  # 安装toolkit  以arm为例
  chmod +x Ascend-cann-toolkit_${version}_linux-aarch64.run
  ./Ascend-cann-toolkit_${version}_linux-aarch64.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

#### 3.2 安装kernel

- 检查包

| 包名                                       |
| ------------------------------------------ |
| Ascend-cann-kernels_${version}_linux.run |

  - 根据芯片型号选择对应的安装包(310B)

- 安装
  ```bash
  chmod +x Ascend-cann-kernels-*_${version}_linux.run
  ./Ascend-cann-kernels_${version}_linux.run --install
  ```

#### 3.3 安装加速库

- 检查包

  | 包名（其中`${version}`为实际版本）            |
  | -------------------------------------------- |
  | Ascend-cann-nnal_${version}_linux-aarch64.run |

- 安装
    ```shell
    chmod +x Ascend-cann-nnal_*_linux-*.run
    ./Ascend-cann-nnal_*_linux-*.run --install 
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```


### 4. 安装PytorchAdapter

先安装torch 再安装torch_npu

#### 4.1 安装torch

- 下载

  | 包名                                         |
  | -------------------------------------------- |
  | torch-2.1.0-cp310-cp310-linux_aarch64.whl     |

  - 根据所使用的环境中的python版本以及cpu类型，选择对应版本的开源torch安装包。

- 安装
  ```bash
  # 安装torch 2.1.0 的python 3.10 的arm版本为例
  pip install torch-2.1.0-cp310-cp310-linux_aarch64.whl
  ```

#### 4.2 安装torch_npu

- 检查包

| 包名                        |
| --------------------------- |
| torch_npu-2.1.0.post13-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl |
| ...                         |

- 安装选择与torch版本以及python版本一致的npu_torch版本

```bash
# 安装 torch_npu，以 torch 2.1.0，python 3.10 的版本为例
pip install torch_npu-2.1.0.post13-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### 5. 安装模型仓

使用编译好的包进行安装
  - 下载编译好的包[链接](https://mindie.obs.cn-north-4.myhuaweicloud.com/artifact/ATB-Models/2.2.T10/Ascend-mindie-atb-models_2.2.T10_linux-aarch64_py310_torch2.1.0-abi0.tar.gz)

    | 包名                                                         |
    | ------------------------------------------------------------ |
    | Ascend-mindie-atb-models_2.2.T10_linux-aarch64_py310_torch2.1.0-abi0.tar.gz |

  - 将文件放置在\${working_dir}路径下
  - 解压
    ```shell
    cd ${working_dir}
    mkdir MindIE-LLM
    cd MindIE-LLM
    tar -zxvf ../Ascend-mindie-atb-models_*_linux-*_torch*-abi*.tar.gz
    ```
  - 安装atb_llm whl包
    ```
    cd ${working_dir}/MindIE-LLM
    # 首次安装
    pip install atb_llm-0.0.1-py3-none-any.whl
    # 更新
    pip install atb_llm-0.0.1-py3-none-any.whl --force-reinstall
    ```

### 6. 安装开源软件依赖

- 默认依赖路径：${working_dir}/MindIE-LLM/requirements/requirements.txt
- 开源软件依赖请使用下述命令进行安装：
  ```bash
  pip install -r requirements.txt
  ```

## 纯模型推理

### 对话测试
进入llm_model路径

```shell
cd $ATB_SPEED_HOME_PATH
```

执行对话测试
-非量化场景
```shell
python   -m examples.run_fa_edge \
         --model_path ${权重路径} \
         --input_text 'What is deep learning?' \
         --max_output_length 20 \
         --is_chat_model \
```
## 声明
- 本代码仓提到的数据集和模型仅作为示例,这些数据集和模型仅供您用于非商业目的,如您使用这些数据集和模型来完成示例,请您特别注意应遵守对应数据集和模型的License,如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中,发现任何问题(包括但不限于功能问题、合规问题),请在本代码仓提交issue,我们将及时审视并解答。