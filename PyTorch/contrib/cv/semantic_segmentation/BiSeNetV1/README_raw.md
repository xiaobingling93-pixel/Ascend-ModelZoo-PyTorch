<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
</div>
<br />

[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

Documentation: https://mmsegmentation.readthedocs.io/

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The master branch works with **PyTorch 1.3 to 1.6**.

![demo image](resources/seg_demo.gif)

### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**

  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.10.0 was released in 01/01/2021.
Please refer to changelog.md for details and release history.

## Benchmark and model zoo

Results and models are available in the model zoo.

Supported backbones:

- [x] ResNet
- [x] ResNeXt
- [x] HRNet
- [x] ResNeSt
- [x] MobileNetV2
- [x] MobileNetV3

Supported methods:

- [x] FCN
- [x] PSPNet
- [x] DeepLabV3
- [x] PSANet
- [x] DeepLabV3+
- [x] UPerNet
- [x] NonLocal Net
- [x] EncNet
- [x] CCNet
- [x] DANet
- [x] APCNet
- [x] GCNet
- [x] DMNet
- [x] ANN
- [x] OCRNet
- [x] Fast-SCNN
- [x] Semantic FPN
- [x] PointRend
- [x] EMANet
- [x] DNLNet
- [x] CGNet
- [x] Mixed Precision (FP16) Training

## Installation

Please refer to INSTALL.md for installation and dataset preparation.

## Get Started

Please see getting_started.md for the basic usage of MMSegmentation.
There are also tutorials for adding new dataset, designing data pipeline, and adding new modules.

A Colab tutorial is also provided. You may preview the notebook here or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/) on Colab.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to CONTRIBUTING.md for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.
