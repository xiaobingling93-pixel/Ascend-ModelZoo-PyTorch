class DeepLabV3(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  backbone : __torch__.torchvision.models._utils.IntermediateLayerGetter
  classifier : __torch__.torchvision.models.segmentation.deeplabv3.DeepLabHead
  aux_classifier : __torch__.torchvision.models.segmentation.fcn.FCNHead
  def forward(self: __torch__.torchvision.models.segmentation.deeplabv3.DeepLabV3,
    x: Tensor):
    aux_classifier = self.aux_classifier
    classifier = self.classifier
    backbone = self.backbone
    _0 = ops.prim.NumToTensor(torch.size(x, 2))
    _1 = int(_0)
    _2 = int(_0)
    _3 = ops.prim.NumToTensor(torch.size(x, 3))
    _4 = int(_3)
    _5 = int(_3)
    _6, _7, = (backbone).forward(x, )
    _8 = torch.upsample_bilinear2d((classifier).forward(_6, ), [_2, _5], False, None)
    _9 = torch.upsample_bilinear2d((aux_classifier).forward(_7, ), [_1, _4], False, None)
    return _8
class DeepLabHead(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torchvision.models.segmentation.deeplabv3.ASPP
  __annotations__["1"] = __torch__.torch.nn.modules.conv.___torch_mangle_164.Conv2d
  __annotations__["2"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_165.BatchNorm2d
  __annotations__["3"] = __torch__.torch.nn.modules.activation.___torch_mangle_166.ReLU
  __annotations__["4"] = __torch__.torch.nn.modules.conv.___torch_mangle_167.Conv2d
  def forward(self: __torch__.torchvision.models.segmentation.deeplabv3.DeepLabHead,
    argument_1: Tensor) -> Tensor:
    _4 = getattr(self, "4")
    _3 = getattr(self, "3")
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _10 = (_1).forward((_0).forward(argument_1, ), )
    _11 = (_4).forward((_3).forward((_2).forward(_10, ), ), )
    return _11
class ASPP(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  convs : __torch__.torch.nn.modules.container.ModuleList
  project : __torch__.torch.nn.modules.container.___torch_mangle_163.Sequential
  def forward(self: __torch__.torchvision.models.segmentation.deeplabv3.ASPP,
    argument_1: Tensor) -> Tensor:
    project = self.project
    convs = self.convs
    _4 = getattr(convs, "4")
    convs0 = self.convs
    _3 = getattr(convs0, "3")
    convs1 = self.convs
    _2 = getattr(convs1, "2")
    convs2 = self.convs
    _1 = getattr(convs2, "1")
    convs3 = self.convs
    _0 = getattr(convs3, "0")
    _12 = [(_0).forward(argument_1, ), (_1).forward(argument_1, ), (_2).forward(argument_1, ), (_3).forward(argument_1, ), (_4).forward(argument_1, )]
    input = torch.cat(_12, 1)
    return (project).forward(input, )
class ASPPConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.conv.___torch_mangle_146.Conv2d
  __annotations__["1"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_147.BatchNorm2d
  __annotations__["2"] = __torch__.torch.nn.modules.activation.___torch_mangle_148.ReLU
  def forward(self: __torch__.torchvision.models.segmentation.deeplabv3.ASPPConv,
    argument_1: Tensor) -> Tensor:
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _13 = (_1).forward((_0).forward(argument_1, ), )
    return (_2).forward(_13, )
class ASPPPooling(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d
  __annotations__["1"] = __torch__.torch.nn.modules.conv.___torch_mangle_157.Conv2d
  __annotations__["2"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_158.BatchNorm2d
  __annotations__["3"] = __torch__.torch.nn.modules.activation.___torch_mangle_159.ReLU
  def forward(self: __torch__.torchvision.models.segmentation.deeplabv3.ASPPPooling,
    argument_1: Tensor) -> Tensor:
    _3 = getattr(self, "3")
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _14 = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _15 = int(_14)
    _16 = ops.prim.NumToTensor(torch.size(argument_1, 3))
    _17 = int(_16)
    _18 = (_1).forward((_0).forward(argument_1, ), )
    _19 = torch.upsample_bilinear2d((_3).forward((_2).forward(_18, ), ), [_15, _17], False, None)
    return _19
