# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
import torch

from MGN.opt import opt
from MGN.data import Data
from MGN.network import MGN
from MGN.utils.metrics import mean_ap, cmc, re_ranking

from om_executor import OMExcutor


class Convertor(OMExcutor):
    def __init__(self, data):
        super().__init__(data)


    def pth2onnx(self, pt_file_path, onnx_file_path, batch_size):
        model = MGN()
        model = model.to('cpu')
        model.load_state_dict(torch.load(pt_file_path, map_location=torch.device('cpu')))
        model.eval()
        input_names = ["image"]
        output_names = ["features"]
        dynamic_axes = {'image': {0: '-1'}, 'features': {0: '-1'}}
        dummy_input = torch.randn(batch_size, 3, 384, 128)
        torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names,
                            dynamic_axes=dynamic_axes, output_names=output_names,
                            opset_version=11, verbose=True)
        print("Convert to ONNX model file SUCCESS!")


if __name__ == '__main__':
    data = Data()
    mgn_convertor = Convertor(data)
    print("start convert to onnx")
    model_pt_file = os.path.join(opt.model_path, opt.model_weight_file)
    model_onnx_file = os.path.join(opt.model_path, opt.onnx_file)
    mgn_convertor.pth2onnx(model_pt_file, model_onnx_file, opt.batchonnx)