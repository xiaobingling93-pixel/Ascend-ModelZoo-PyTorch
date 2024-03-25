# SSD-MobileNetV1 for Pytorch

# 目录

- [简介](#简介)
    - [模型介绍](#模型介绍)
    - [代码实现](#代码实现)

# 简介

## 模型介绍

MobileNetV1是基于深度级可分离卷积构建的网络。 MobileNetV1将标准卷积拆分为了两个操作：深度卷积和逐点卷积 。
SSD是一种one-stage的目标检测框架。SSD_MobileNetV1使用MobileNetV1提取有效特征，之后SSD通过得到的特征图的信息进行检测。

## 代码实现

- 参考实现：

  ```
  url=https://github.com/qfgaohao/pytorch-ssd
  commit_id=7a839cbc8c3fb39679856b4dc42a1ab19ec07581
  ```
