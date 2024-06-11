/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/script.h>
#include <torch/torch.h>

#include <vector>


at::Tensor QuantizeTensorPlaceholder(at::Tensor x, at::Tensor scale, at::Tensor offset)
{
    auto quant_x = x * scale;
    quant_x = quant_x + torch::broadcast_tensors({offset, quant_x})[0];
    quant_x = quant_x.round();
    quant_x = quant_x.clamp(-128, 127);
    return quant_x;
}

at::Tensor QuantizeFloatPlaceholder(at::Tensor x, double scale, double offset)
{
    auto quant_x = x * scale + offset;
    quant_x = quant_x.round();
    quant_x = quant_x.clamp(-128, 127);
    return quant_x;
}

at::Tensor DequantizeTensorPlaceholder(at::Tensor x, at::Tensor scale)
{
    auto fp_x = x;
    auto round_x = fp_x.round();
    auto dequant_x = round_x.clamp(-128, 127);
    return dequant_x;
}

at::Tensor DequantizeFloatPlaceholder(at::Tensor x, double scale)
{
    auto fp_x = x;
    auto round_x = fp_x.round();
    auto dequant_x = round_x.clamp(-128, 127);
    return dequant_x;
}

at::Tensor QuantConvolutionPlaceholder(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias, at::IntArrayRef stride,
                             at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
                             int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32)
{
    auto fp_weight = weight.to(torch::kFloat);
    if (bias.has_value()) {
        auto fp_bias = bias.value().to(torch::kFloat);
        auto output = torch::_convolution(input, fp_weight, fp_bias, stride, padding, dilation, transposed, output_padding,
                                          groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
        return output;
    }else{
        auto output = torch::_convolution(input, fp_weight, bias, stride, padding, dilation, transposed, output_padding, groups,
                                          benchmark, deterministic, cudnn_enabled, allow_tf32);
        return output;
    }

}

at::Tensor QuantLinearPlaceholder(at::Tensor input, at::Tensor weight, c10::optional<at::Tensor> bias)
{
    auto fp_weight = weight.to(torch::kFloat);
    if (bias.has_value()) {
        auto fp_bias = bias.value().to(torch::kFloat);
        auto output = torch::linear(input, fp_weight, fp_bias);
        return output;
    }else{
        auto output = torch::linear(input, fp_weight,bias);
        return output;
    }
}

// register torchscript quant ops schema to Pytorch
TORCH_LIBRARY_FRAGMENT(MindIE, m) {
m.def(TORCH_SELECTIVE_SCHEMA("MindIE::quantize.tensor(Tensor x, Tensor scale, Tensor offset) -> Tensor"));
m.def(TORCH_SELECTIVE_SCHEMA("MindIE::quantize.float(Tensor x,float scale, float offset) -> Tensor"));
m.def(TORCH_SELECTIVE_SCHEMA("MindIE::dequantize.tensor(Tensor x, Tensor scale) -> Tensor"));
m.def(TORCH_SELECTIVE_SCHEMA("MindIE::dequantize.float(Tensor x, float scale) -> Tensor"));
m.def(TORCH_SELECTIVE_SCHEMA("MindIE::quant_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding,\n"
                             "        int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic,\n"
                             "        bool cudnn_enabled, bool allow_tf32) -> Tensor"));
m.def(TORCH_SELECTIVE_SCHEMA("MindIE::quant_linear(Tensor input, Tensor weight, Tensor? bias = None) -> (Tensor)"));
}

// register CPU kernel function for all_reduce
TORCH_LIBRARY_IMPL(MindIE, CPU, m) {
m.impl(
        TORCH_SELECTIVE_NAME("MindIE::quantize.tensor"),
        TORCH_FN(QuantizeTensorPlaceholder)
);
m.impl(
        TORCH_SELECTIVE_NAME("MindIE::quantize.float"),
        TORCH_FN(QuantizeFloatPlaceholder)
);
m.impl(
        TORCH_SELECTIVE_NAME("MindIE::dequantize.tensor"),
        TORCH_FN(DequantizeTensorPlaceholder)
);
m.impl(
        TORCH_SELECTIVE_NAME("MindIE::dequantize.float"),
        TORCH_FN(DequantizeFloatPlaceholder)
);
m.impl(
        TORCH_SELECTIVE_NAME("MindIE::quant_convolution"),
        TORCH_FN(QuantConvolutionPlaceholder)
);
m.impl(
        TORCH_SELECTIVE_NAME("MindIE::quant_linear"),
        TORCH_FN(QuantLinearPlaceholder)
);
}
