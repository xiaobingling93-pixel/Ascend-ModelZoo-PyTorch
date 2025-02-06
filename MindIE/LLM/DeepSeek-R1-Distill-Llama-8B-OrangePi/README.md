## 约束条件
* 在20t24g 香橙派aipro上部署DeepSeek-R1-Distill-Llama-8B模型
* 需要修改权重目录下的config.json文件，"torch_dtype"字段改为"float16", "max_position_embedding"字段改为8192
* 由于此硬件为单卡，仅支持TP=1

## 权重

**权重下载**

- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/tree/main)

## 新建环境

### 1.1 安装CANN
- 详细信息可参见[昇腾社区CANN软件](https://www.hiascend.com/software/cann)
- 安装顺序：先安装toolkit 再安装kernel

#### 1.1.1 安装toolkit

- 下载

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

#### 1.1.2 安装kernel

- 下载

| 包名                                       |
| ------------------------------------------ |
| Ascend-cann-kernels-*_${version}_linux.run |

  - 根据芯片型号选择对应的安装包

- 安装
  ```bash
  chmod +x Ascend-cann-kernels-*_${version}_linux.run
  ./Ascend-cann-kernels-*_${version}_linux.run --install
  ```

#### 1.1.3 安装加速库
- 下载加速库
  - [下载链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261918053?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)。

  | 包名（其中`${version}`为实际版本）            |
  | -------------------------------------------- |
  | Ascend-cann-nnal_${version}_linux-aarch64.run |

  - 将文件放置在\${working_dir}路径下

- 安装
    ```shell
    chmod +x Ascend-cann-nnal_*_linux-*.run
    ./Ascend-cann-nnal_*_linux-*.run --install --install-path=${working_dir}
    source ${working_dir}/nnal/atb/set_env.sh
    ```
- 可以使用以下指令查看abi是0还是1
    ```shell
    python -c "import torch; print(torch.compiled_with_cxx11_abi())"
    ```
    - 若输出结果为True表示abi1，False表示abi0

### 1.2 安装PytorchAdapter

先安装torch 再安装torch_npu

#### 1.2.1 安装torch

- 下载

  | 包名                                         |
  | -------------------------------------------- |
  | torch-2.1.0-cp310-cp10-linux_aarch64.whl     |

  - 根据所使用的环境中的python版本以及cpu类型，选择对应版本的torch安装包。

- 安装
  ```bash
  # 安装torch 2.1.0 的python 3.10 的arm版本为例
  pip install torch-2.1.0-cp310-cp310-linux_aarch64.whl
  ```

#### 1.2.2 安装torch_npu

[下载PyTorch Adapter](https://www.hiascend.com/developer/download/community/result?module=pt)，安装方法：

| 包名                        |
| --------------------------- |
| pytorch_v2.1.0_py38.tar.gz |
| pytorch_v2.1.0_py39.tar.gz |
| pytorch_v2.1.0_py310.tar.gz |
| ...                         |

- 安装选择与torch版本以及python版本一致的npu_torch版本

```bash
# 安装 torch_npu，以 torch 2.1.0，python 3.10 的版本为例
tar -zxvf pytorch_v2.1.0_py310.tar.gz
pip install torch*_aarch64.whl
```
### 1.3 安装开源软件依赖
| 默认依赖                 | [requirement.txt](./requirements.txt)           |
- 开源软件依赖请使用下述命令进行安装：
  ```bash
  pip install -r ./requirements.txt
  ```

### 1.4 安装模型仓
- 使用编译好的包进行安装
  - 下载编译好的包
    - [下载链接](https://www.hiascend.com/developer/download/community/result?module=ie+pt+cann)

    | 包名                                                         |
    | ------------------------------------------------------------ |
    | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torch1.11.0-abi0.tar.gz |
    | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torch2.1.0-abi1.tar.gz |

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

## 模型推理

### 对话测试
进入llm_model路径

```shell
cd $ATB_SPEED_HOME_PATH
```

执行对话测试

```shell
python   -m examples.run_fa_edge \
         --model_path ${权重路径} \
         --input_text 'What is deep learning?' \
         --max_output_length 20 \
```