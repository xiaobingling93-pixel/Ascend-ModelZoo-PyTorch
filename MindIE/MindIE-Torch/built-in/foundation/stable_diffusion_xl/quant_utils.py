import torch
import torch.nn as nn
import numpy as np


class Quantize(nn.Module):
    def __init__(self, scale, offset):
        super(Quantize, self).__init__()
        self.offset = offset
        self.scale = 1 / scale

    def forward(self, x):
        x = torch.ops.MindIE.quantize(x, self.scale.item(), self.offset.item())
        return x


class DeQuantize(nn.Module):
    def __init__(self, scale):
        super(DeQuantize, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = torch.ops.MindIE.dequantize(x, self.scale)
        return x


class QuantConvModule(nn.Module):
    def __init__(self, layer, input_scale, input_offset, quant_weight, weight_scale, deq_scale):
        super(QuantConvModule, self).__init__()
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.quant = Quantize(scale=input_scale, offset=input_offset)
        self.weight = torch.nn.Parameter(quant_weight, requires_grad=False)
        self.set_bias = False
        self.layer = layer
        if self.layer.bias is not None:
            self.bias = torch.nn.Parameter(torch.round(self.layer.bias / torch.squeeze(input_scale) / torch.squeeze(
                weight_scale)).to(torch.int32), requires_grad=False)
        else:
            self.bias = None
        self.de_quant = DeQuantize(deq_scale)

    def forward(self, x, scale: float = 1.0):
        x = self.quant(x)
        x = torch.ops.MindIE.quant_convolution(x, self.weight, self.bias, self.layer.stride, self.layer.padding,
                                               self.layer.dilation, self.layer.transposed, self.layer.output_padding,
                                               self.layer.groups, False, False,
                                               False, False)
        x = self.de_quant(x)
        return x


class QuantLinearModule(nn.Module):
    def __init__(self, layer, input_scale, input_offset, quant_weight, weight_scale, deq_scale):
        super(QuantLinearModule, self).__init__()
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.quant = Quantize(scale=input_scale, offset=input_offset)
        self.layer = layer
        self.weight = torch.nn.Parameter(quant_weight, requires_grad=False)
        self.set_bias = False
        if self.layer.bias is not None:
            self.bias = torch.nn.Parameter(torch.round(self.layer.bias / torch.squeeze(input_scale) / torch.squeeze(
                weight_scale)).to(torch.int32), requires_grad=False)
        else:
            self.bias = None
        self.de_quant = DeQuantize(deq_scale)

    def forward(self, x, scale: float = 1.0):
        x = self.quant(x)
        x = torch.ops.MindIE.quant_linear(x, self.weight, self.bias)
        x = self.de_quant(x)
        return x


def modify_model(model, input_scale_dict, input_offset_dict, weight_scale_dict, weight_offset_dict, quant_weight_dict):
    torch.ops.load_library("./quant/build/libquant_ops.so")
    for name, layer in model.named_modules():
        if name in input_scale_dict:
            if quant_weight_dict[name] is None:
                continue
            input_scale = input_scale_dict[name] if input_scale_dict[name] is not None else torch.Tensor([1.])
            input_offset = input_offset_dict[name] if input_offset_dict[name] is not None else torch.Tensor([0.])
            quant_weight = quant_weight_dict[name].to(torch.int8)
            weight_scale = weight_scale_dict[name]

            x_scale = np.array(input_scale) * np.array(weight_scale)
            packed_weight_np_data = x_scale.squeeze()
            float32_scale_deq = np.array(packed_weight_np_data, np.float32)
            uint32_scale_deq = np.frombuffer(float32_scale_deq, np.uint32)
            uint64_result = np.zeros(float32_scale_deq.shape, np.int64)
            # per-tensor
            if len(uint64_result.shape) == 0:
                uint64_result = np.expand_dims(uint64_result, axis=0)
            uint64_result |= np.int64(uint32_scale_deq)

            deq_scale = torch.Tensor(uint64_result).to(torch.int64)
            if isinstance(layer, nn.Conv2d):
                quant_module = QuantConvModule(layer, input_scale, input_offset, quant_weight, weight_scale, deq_scale)
            elif isinstance(layer, nn.Linear):
                quant_module = QuantLinearModule(layer, input_scale, input_offset, quant_weight, weight_scale,
                                                 deq_scale)
            else:
                continue

            submodules, layer_name = name.split('.')[:-1], name.split('.')[-1]
            setattr(model.get_submodule('.'.join(submodules)), layer_name, quant_module)
            print(f'converter layer {name} from {layer.__class__.__name__} to {quant_module.__class__.__name__} succ')
    return model
