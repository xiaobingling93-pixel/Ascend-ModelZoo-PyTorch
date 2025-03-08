#!/usr/bin/env python
# coding=utf-8
import os
import torch
from torch import distributed as dist
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from utils.parallel_mgr import ParallelConfig, init_parallel_env, finalize_parallel_env, get_sequence_parallel_rank

# 如果使用npu进行量化需开启二进制编译，避免在线编译算子
torch.npu.set_compile_mode(jit_compile=False)

use_fa3 = True
world_size = int(os.getenv('WORLD_SIZE', 1))
if world_size > 1:
    sp_degree = world_size // 2
    parallel_config = ParallelConfig(sp_degree=sp_degree, use_cfg_parallel=True,
                                     world_size=world_size)
    init_parallel_env(parallel_config)
rank = dist.get_rank()

model_path = '/home/Open-Sora-Plan-v1.2.0/93x720p'  # 原始浮点模型路径
dev_id = 8
model = OpenSoraT2V.from_pretrained(model_path, cache_dir="../cache_dir",
                                    low_cpu_mem_usage=False, device_map=None,
                                    torch_dtype=torch.bfloat16).to("npu")
calib_datas = torch.load(f"/home/quant_model/calib_datas.pt", map_location='cpu')
for calib_data in calib_datas:
    for i, data in enumerate(calib_data):
        if torch.is_tensor(data):
            calib_data[i] = data.npu()

# 调用fa_quant之后默认开启FA量化，fa_amp可设置自动回退层数
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=None,
    dev_type='npu',
    dev_id=rank,
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
    is_dynamic=True,
).fa_quant(fa_amp=0)

calibrator = Calibrator(model, quant_config, calib_data=calib_datas, disable_level='L0',
                        torch_dtype=torch.bfloat16)
calibrator.run()

# fa3需要按不同卡去保存权重，不用fa3则使用默认保存
if use_fa3:
    calibrator.save('/home/quant_model', safetensors_name=f'quant_model_weight_w8a8_dynamic_{rank}.safetensors',
                    save_type=["safe_tensor"],
                    json_name=f'quant_model_description_w8a8_dynamic_{rank}.json')  # "safe_tensor"对应safetensors格式权重，"numpy"对应npy格式权重
elif rank == 0:
    calibrator.save('/home/quant_model', save_type=["safe_tensor"])
