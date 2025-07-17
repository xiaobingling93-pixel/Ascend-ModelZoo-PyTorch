# OpenFold_for_PyTorch

- [OpenFold\_for\_PyTorch](#OpenFold_for_PyTorch)
- [概述](#概述)
  - [简述](#简述)
- [准备训练环境](#准备训练环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [开始训练](#开始训练)
  - [训练模型](#训练模型)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
- [公网地址说明](#公网地址说明)

# 概述

## 简述
在生物学中，结构和功能密不可分。因此，理解生物系统的机制、工程设计及其影响方式，就意味着需要了解和理解它们的结构。该联盟正在开发基于人工智能的先进蛋白质建模工具，能够以原子级精度预测分子结构，首次以开源形式将这种精度水平用于研究和商业应用。世界各地的研究人员将能够使用、改进并贡献这种“预测分子显微镜”。

- 参考实现：

  ```
  url=https://github.com/aqlaboratory/openfold.git
  commit_id=e8d355874c3cc767e56af983d4e9a5190918eb6c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch
  code_path=built-in/PyTorch/built-in/others
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version | 依赖python版本                                 |
  |:-------------:| :----------------------------------------------------------: |
  |  PyTorch 2.1  | python=3.9 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

  安装其他依赖
  ```
  pip install git+https://github.com/NVIDIA/dllogger.git
  pip install torch==2.1.0

  wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz
  tar xvfz hhsuite-3.3.0-AVX2-Linux.tar.gz
  export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"

  pip install git+https://github.com/TimoLassmann/kalign.git

  wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
  tar xvfz mmseqs-linux-avx2.tar.gz
  export PATH=$(pwd)/mmseqs/bin/:$PATH
  ```

- 构建安装openfold。

  ```
  bash scripts/install_third_party_dependencies.sh
  python setup.py install
  ```

## 准备数据集
    对于本指南，我们假设 OpenFold 代码库位于$OF_DIR

1. 安装aws。

    ```
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    ```

2. 获取数据集。

   要获取与 OpenFold 的原始 PDB 训练集相对应的所有比对及其 mmCIF 3D 结构，您可以运行以下命令：

   ```
    mkdir -p alignment_data/alignment_dir_roda
    aws s3 cp s3://openfold/pdb/ alignment_data/alignment_dir_roda/ --recursive --no-sign-request

    mkdir pdb_data
    aws s3 cp s3://openfold/pdb_mmcif.zip pdb_data/ --no-sign-request
    aws s3 cp s3://openfold/duplicate_pdb_chains.txt . --no-sign-request
    unzip pdb_mmcif.zip -d pdb_data
   ```
   
   嵌套对齐目录结构还不完全符合 OpenFold 的期望，因此您可以运行flatten_roda.sh脚本将它们转换为正确的格式：
    ```
    bash $OF_DIR/scripts/flatten_roda.sh alignment_data/alignment_dir_roda alignment_data/
    ```

    之后，可以安全地删除旧目录：
    ```
    rm -r alignment_data/alignment_dir_roda
    ```

3. 生成数据切片

     ```
    python $OF_DIR/scripts/alignment_db_scripts/create_alignment_db_sharded.py \
        alignment_data/alignments \
        alignment_data/alignment_dbs \
        alignment_db \
        --n_shards 10 \
        --duplicate_chains_file pdb_data/duplicate_pdb_chains.txt
     ```

    作为可选检查，您可以运行以下命令，该命令应返回634434：
    ```
    grep "files" alignment_data/alignment_dbs/alignment_db.index | wc -l
    ```

4. 下载pdb_cache

    ```
    aws s3 cp s3://openfold/data_caches/ pdb_data/ --recursive --no-sign-request
    ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   - 8卡训练

     启动8卡训练。

     ```
     bash test/train_openfold_8p.sh --data_path=xxxx  # openfold 8p training
     ```

   - 8卡验证精度

     启动8卡验证（需跑完训练后进行验证）。

     ```
     bash test/val_openfold_8p.sh --data_path=xxxx --val_alignment_dir=xxxx --val_data_dir=xxxx  # openfold 8p validation
     ```


   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //训练数据集路径，必填
   --val_data_dir                           //包含验证集mmCIF文件的路径，验证阶段必填
   --val_alignment_dir                      //包含验证集alignments文件的路径，验证阶段必填
   --max_epochs                             //重复训练次数，默认为1
   ```
   
   训练完成后，权重文件保存在output/checkpoints下，并输出模型训练时间。
   
   接着运行验证脚本，输出模型训练后的精度验证数据。

# 训练结果展示

**表 2**  OpenFold训练&验证结果展示表

| NAME     | MODE | training_time| val/loss  | Torch_Version |
| :-----:  | :---:  | :--------:| :--------:| :------: |
| 8p-竞品A  | bf16 |  1:38:03   |  78.30   |    2.1 |
| 8p-Atlas 900 A2 PODc | bf16 |  2:37:31   |  79.32   |      2.1      |

# 版本说明

## 变更

2025.04.09：首次提交。

2025.04.12：适配NPU，新增训练和验证脚本。

## FAQ

无。

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
