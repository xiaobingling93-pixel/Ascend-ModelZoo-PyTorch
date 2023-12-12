import os
import sys
import time
import math
import numpy as np
import torch
import torch_aie
from torch_aie import _enums
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding
from diffusers import StableDiffusionPipeline
from config import NpuConfig
from pt_background_runtime_np import BackgroundRuntime, RuntimeIOInfo

class UnetExport(torch.nn.Module):
    def __init__(self, model):
        super(UnetExport, self).__init__()
        self.unet_model = model

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet_model(sample, timestep, encoder_hidden_states)[0]

def replace_unet_torch_aie():
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs("models")
    device_0, device_1 = 2, None
    torch_aie.set_device(device_0)
    model_base = "runwayml/stable-diffusion-v1-5"
    if NpuConfig.use_parallel_inferencing:
        batch_size = 1
        device_1 = 3
        unet_path = os.path.join(cur_dir_path, "models", "unet_aie_compile_bs1.pt")
    else:
        batch_size = 2
        unet_path = os.path.join(cur_dir_path, "models", "unet_aie_compile_bs2.pt")
    
    def torch_aie_unet(self, x, timesteps = None, context = None, y = None, **kwargs):
        if not NpuConfig.compiled_unet_model:
            if not os.path.exists(unet_path):
                pipe = StableDiffusionPipeline.from_pretrained(model_base).to("cpu")
                in_channels = pipe.unet.config.out_channels
                sample_size = pipe.unet.config.sample_size
                encoder_hidden_size = pipe.text_encoder.config.hidden_size
                max_position_embeddings = pipe.text_encoder.config.max_position_embeddings
                dummy_input = (
                    torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
                    torch.ones([1], dtype=torch.int64),
                    torch.ones(
                        [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
                    ),
                )
                unet = UnetExport(pipe.unet)
                model = torch.jit.trace(unet, dummy_input)
                unet_input_info = [
                    torch_aie.Input((batch_size, in_channels, sample_size, sample_size), dtype=torch_aie.dtype.FLOAT),
                    torch_aie.Input((1,), dtype=torch_aie.dtype.INT64),
                    torch_aie.Input((batch_size, max_position_embeddings, encoder_hidden_size),
                                    dtype=torch_aie.dtype.FLOAT)]
                compiled_unet_model = torch_aie.compile(model, inputs=unet_input_info,
                                                            allow_tensor_replace_int=True,
                                                            require_full_compilation=True,
                                                            truncate_long_and_double=True,
                                                            soc_version="Ascend910B3",
                                                            precision_policy=_enums.PrecisionPolicy.FP16,
                                                            optimization_level=1
                                                            )
                torch.jit.save(compiled_unet_model, unet_path)
                NpuConfig.compiled_unet_model = compiled_unet_model
            else:
                NpuConfig.compiled_unet_model = torch.jit.load(unet_path).eval()
            if NpuConfig.use_parallel_inferencing:
                NpuConfig.unet_bg = BackgroundRuntime.clone(device_1, unet_path, runtime_info)
        
        if NpuConfig.use_parallel_inferencing:
            context, context_2 = context.chunk(2)
            x, x_2 = x.chunk(2)
            NpuConfig.unet_bg.infer_asyn(
                x_2.numpy(),
                timesteps[0][None].numpy().astype(np.int64),
                context_2.numpy(),
            )
        noise_pred = NpuConfig.compiled_unet_model(x.to(f'npu:{device_0}'), 
                                                   timesteps[0][None].type(torch.int64).to(f'npu:{device_0}'),
                                                   context.to(f'npu:{device_0}')).to('cpu')
        if NpuConfig.use_parallel_inferencing:
            noise_pred_text = torch.from_numpy(NpuConfig.unet_bg.wait_and_get_outputs()[0])
            noise_pred = torch.cat([noise_pred, noise_pred_text])
        return noise_pred
    UNetModel.forward = torch_aie_unet
            