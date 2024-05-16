import numpy as np
import torch
import mindietorch
from models import DiT
from background_runtime import BackgroundRuntime, RuntimeIOInfo

class MindIEDiT(DiT):
    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        device = self.device
        if self.parallel:
            combined, combined_2 = combined.chunk(2)
            t, t_2 = t.chunk(2)
            y, y_2 = y.chunk(2)
            self.bg.infer_asyn([
                combined_2.numpy().astype(np.float32),
                t_2.numpy().astype(np.int64),
                y_2.numpy().astype(np.int64)
            ])
        with mindietorch.npu.stream(self.stream):
            model_out_npu = self.model_npu(combined.to(f"npu:{device}"),
                                           t.to(f"npu:{device}"),
                                           y.to(f"npu:{device}"))
            self.stream.synchronize()
        model_out = model_out_npu.to("cpu")

        if self.parallel:
            model_out_2 = torch.from_numpy(self.bg.wait_and_get_outputs()[0])
            model_out = torch.cat([model_out, model_out_2])
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def set_npu_model_stream(self, parallel, device, image_size, mindie_model_path):
        latent_size = image_size // 8
        self.device, device_2 = device, device + 1
        self.stream = mindietorch.npu.Stream(f"npu:{self.device}")
        self.parallel = parallel
        self.model_npu = torch.jit.load(mindie_model_path).eval()
        if parallel:
            runtime_info = RuntimeIOInfo(
                input_shapes=[(1, 4, latent_size, latent_size), (1,), (1,)],
                input_dtypes=[np.float32, np.int64, np.int64],
                output_shapes=[(1, 8, latent_size, latent_size)],
                output_dtypes=[np.float32]
            )
            self.bg = BackgroundRuntime(device_2, mindie_model_path, runtime_info)
        print('success init')
    
    def end_asyn(self):
        self.bg.stop()

def DiT_XL_2(**kwargs):
    return MindIEDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2
}