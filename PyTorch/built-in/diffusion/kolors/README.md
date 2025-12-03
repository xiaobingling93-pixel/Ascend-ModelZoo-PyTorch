# Kolors
# 目录
- [Kolors](#Kolors)
- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备环境](#准备环境)
  - [安装模型环境](#安装模型环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [获取预训练模型](#获取预训练模型)
- [快速开始](#快速开始)
  - [推理](#推理)


# 简介
## 模型介绍

**Kolors** 是基于潜在扩散的大规模文本生成图像模型，由快手 Kolors 团队开发。该模型在数十亿的文本-图像对上进行训练，在视觉质量、复杂语义准确性以及中英文字符的文本呈现上，相较于开源和闭源模型展现出显著优势。此外，Kolors 支持中文和英文输入，且在理解和生成中文特定内容方面表现出色。
本仓库主要将Kolors模型的任务迁移到了昇腾NPU上。

## 支持任务列表

本仓已经支持以下模型任务类型

|  模型  | 任务列表 | 是否支持 |
|:----:|:----:|:----:|
| Kolors | 在线推理 | demo |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/Kwai-Kolors/Kolors.git
  commit_id=2de245f97629208f37dd8ac57aa736e92c7b31d1
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/diffusion/kolors
  ```



## 准备环境

### 安装模型环境


  **表 1**  三方库版本支持表

  |     三方库     |  支持版本  |
  |:-----------:|:------:|
  |   PyTorch   | 2.1.0  |


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```python
   cd PyTorch/built-in/diffusion/kolors
   pip install -r requirements.txt                                    # 安装依赖
   ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                

  **表 2**  昇腾软件版本支持表

  | 软件类型   |   支持版本   |
  | :--------: |:--------:|
  | FrameworkPTAdapter |   在研版本   |
  | CANN |   在研版本   |
  | 昇腾NPU固件 |   在研版本   | 
  | 昇腾NPU驱动 | 在研版本 |

### 获取预训练模型

用户可访问huggingface或ModelScope网站获取Kolors模型权重。

## 快速开始

### 推理
支持单机单卡demo版本推理：

```shell
bash test/infer_1p.sh # 运行单卡推理demo
```
需要修改shell脚本中模型文件路径后运行。
