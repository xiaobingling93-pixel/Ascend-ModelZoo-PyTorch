<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
</div>

**News**: We released the technical report on [ArXiv](https://arxiv.org/abs/1906.07155).

Documentation: https://mmdetection.readthedocs.io/

## Introduction

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

The master branch works with **PyTorch 1.3 to 1.6**.
The old v1.x branch works with PyTorch 1.1 to 1.4, but v2.0 is strongly recommended for faster speed, higher performance, better design and more friendly usage.

![demo image](resources/coco_test_12510.jpg)

### Major features

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## License

This project is released under the Apache 2.0 license.

## Changelog

v2.6.0 was released in 1/11/2020.
Please refer to changelog.md for details and release history.
A comparison between v1.x and v2.0 codebases can be found in compatibility.md.

## Benchmark and model zoo

Results and models are available in the model zoo.

Supported backbones:
- [x] ResNet
- [x] ResNeXt
- [x] VGG
- [x] HRNet
- [x] RegNet
- [x] Res2Net

Supported methods:
- [x] RPN
- [x] Fast R-CNN
- [x] Faster R-CNN
- [x] [Mask R-CNN](configs/mask_rcnn)
- [x] Cascade R-CNN
- [x] Cascade Mask R-CNN
- [x] SSD
- [x] RetinaNet
- [x] GHM
- [x] Mask Scoring R-CNN
- [x] Double-Head R-CNN
- [x] Hybrid Task Cascade
- [x] Libra R-CNN
- [x] Guided Anchoring
- [x] FCOS
- [x] RepPoints
- [x] Foveabox
- [x] FreeAnchor
- [x] NAS-FPN
- [x] ATSS
- [x] FSAF
- [x] PAFPN
- [x] Dynamic R-CNN
- [x] PointRend
- [x] CARAFE
- [x] DCNv2
- [x] Group Normalization
- [x] Weight Standardization
- [x] OHEM
- [x] Soft-NMS
- [x] Generalized Attention
- [x] [GCNet](configs/gcnet/README.md)
- [x] Mixed Precision (FP16) Training
- [x] InstaBoost
- [x] GRoIE
- [x] DetectoRS
- [x] Generalized Focal Loss
- [x] CornerNet
- [x] Side-Aware Boundary Localization
- [x] YOLOv3
- [x] PAA
- [x] YOLACT
- [x] CentripetalNet
- [x] VFNet

Some other methods are also supported in projects using MMDetection.

## Installation

Please refer to get_started.md for installation.

## Getting Started

Please see get_started.md for the basic usage of MMDetection.
We provide colab tutorial, and full guidance for quick run with existing dataset and with new dataset for beginners.
There are also tutorials for finetuning models, adding new dataset, designing data pipeline, customizing models, customizing runtime settings and useful tools.

For trouble shooting, please refer to trouble_shooting.md

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to CONTRIBUTING.md for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Contact

This repo is currently maintained by Kai Chen ([@hellock](http://github.com/hellock)), Yuhang Cao ([@yhcao6](https://github.com/yhcao6)), Wenwei Zhang ([@ZwwWayne](https://github.com/ZwwWayne)),
Jiarui Xu ([@xvjiarui](https://github.com/xvjiarui)). Other core developers include Jiangmiao Pang ([@OceanPang](https://github.com/OceanPang)) and Jiaqi Wang ([@myownskyW7](https://github.com/myownskyW7)).
