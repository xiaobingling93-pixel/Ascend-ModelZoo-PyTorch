import numpy as np
import torch
import argparse
import os
import mindietorch
from mindietorch import _enums
from opensora.models.stdit.stdit import STDiT_XL_2
from opensora.models.vae.vae import VideoAutoencoderKL
from transformers import T5EncoderModel

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="save dir"
    )
    parser.add_argument(
        "--encoder_model_path",
        type=str,
        default="./DeepFloyd--t5-v1_1-xxl",
        help="encoder model path"
    )
    parser.add_argument(
        "--dit_model_path",
        type=str,
        default="./OpenSora-v1-HQ-16x512x512.pth",
        help="stdit model path"
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="./sd-vae-ft-ema",
        help="vae model path"
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=4,
        help="vae micro_batch_size"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default='16x512x512',
        choices=['16x256x256', '16x512x512']
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="npu device id"
    )
    return parser.parse_args()

class TextEncoderExport(torch.nn.Module):
    def __init__(self, textencoder_model):
        super(TextEncoderExport, self).__init__()
        self.textencoder_model = textencoder_model
    
    def forward(self, input_ids, attention_mask):
        return self.textencoder_model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      return_dict=False)[0]

def export_textencoder(args, save_dir, batch_size):
    encoder_path = os.path.join(save_dir, "encoder")
    if not os.path.exists(encoder_path):
        os.makedirs(encoder_path, mode=0o640)
    traced_path = os.path.join(encoder_path, "encoder.pt")
    compiled_path = os.path.join(encoder_path, "encoder_compiled.pt")
    model_path = args.encoder_model_path
    max_lenth = 120
    if not os.path.exists(traced_path):
        text_encoder = T5EncoderModel.from_pretrained(model_path, cache_dir="cache_dir", torch_dtype=torch.float).to('cpu')
        dummy_input = (
            torch.ones([batch_size, max_lenth], dtype=torch.int64),
            torch.ones([batch_size, max_lenth], dtype=torch.int64)
        )
        encoder = TextEncoderExport(text_encoder)
        encoder.eval()
        torch.jit.trace(encoder, dummy_input).save(traced_path)
    if not os.path.exists(compiled_path):
        model = torch.jit.load(traced_path).eval()
        compiled_model = mindietorch.compile(
            model,
            inputs=[mindietorch.Input((batch_size, max_lenth),
                                    dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size, max_lenth),
                                    dtype=mindietorch.dtype.INT64)],
            allow_tensor_replace_int=True,
            require_full_compilation=True,
            truncate_long_and_double=True,
            precision_policy=_enums.PrecisionPolicy.FP32,
            soc_version="Ascend910B4",
            optimization_level=0
        )
        torch.jit.save(compiled_model, compiled_path)

class STDiTExport(torch.nn.Module):
    def __init__(self, dit_model):
        super(STDiTExport, self).__init__()
        self.dit_model = dit_model
    
    def forward(self, x, timestep, y, mask):
        return self.dit_model(x, timestep, y, mask)

def export_dit(args, save_dir, batch_size):
    dit_path = os.path.join(save_dir, "dit")
    if not os.path.exists(dit_path):
        os.makedirs(dit_path, mode=0o640)
    resolution = args.resolution
    latent1 = int(resolution.split('x')[1])
    latent2 = int(resolution.split('x')[2])
    height, width = latent1 // 8, latent2 // 8
    traced_path = os.path.join(dit_path, f"dit_{latent1}_{latent2}.pt")
    compiled_path = os.path.join(dit_path, f"dit_{latent1}_{latent2}_compiled.pt")
    model_path = args.dit_model_path

    kwargs = {
        'space_scale': 0.5,
        'time_scale': 1.0,
        'enable_flashattn': False,
        'enable_layernorm_kernel': False,
        'input_size': [16, height, width],
        'in_channels': 4,
        'caption_channels': 4096,
        'model_max_length': 120,
        'dtype': torch.float32,
        'enable_sequence_parallelism': False
    }

    video_lenth = kwargs['input_size'][0]
    in_channels = kwargs['in_channels']
    model_max_length = kwargs['model_max_length']
    caption_channels = kwargs['caption_channels']
    if not os.path.exists(traced_path):
        dit_model = STDiT_XL_2(from_pretrained=model_path, **kwargs)
        dummy_input = (
            torch.ones([batch_size, in_channels, video_lenth, height, width], dtype=torch.float32),
            torch.ones([batch_size,], dtype=torch.int64),
            torch.ones([batch_size, 1, model_max_length, caption_channels], dtype=torch.float32),
            torch.ones([1, model_max_length], dtype=torch.int64)
        )
        dit = STDiTExport(dit_model)
        dit.eval()
        torch.jit.trace(dit, dummy_input).save(traced_path)
    if not os.path.exists(compiled_path):
        model = torch.jit.load(traced_path).eval()
        compiled_model = mindietorch.compile(
            model,
            inputs=[mindietorch.Input((batch_size, in_channels, video_lenth, height, width),
                                    dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((batch_size,),
                                    dtype=mindietorch.dtype.INT64),
                    mindietorch.Input((batch_size, 1, model_max_length, caption_channels),
                                    dtype=mindietorch.dtype.FLOAT),
                    mindietorch.Input((1, model_max_length),
                                    dtype=mindietorch.dtype.INT64)],
            allow_tensor_replace_int=True,
            require_full_compilation=True,
            truncate_long_and_double=True,
            precision_policy=_enums.PrecisionPolicy.FP16,
            soc_version="Ascend910B4",
            optimization_level=0
        )
        torch.jit.save(compiled_model, compiled_path)

class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super(VaeExport, self).__init__()
        self.vae_model = vae_model
    
    def forward(self, latents):
        return self.vae_model.decode(latents)

def export_vae(args, save_dir, batch_size):
    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    resolution = args.resolution
    latent1 = int(resolution.split('x')[1])
    latent2 = int(resolution.split('x')[2])
    height, width = latent1 // 8, latent2 // 8
    traced_path = os.path.join(vae_path, f"vae_{latent1}_{latent2}.pt")
    compiled_path = os.path.join(vae_path, f"vae_{latent1}_{latent2}_compiled.pt")
    model_path = args.vae_model_path
    micro_batch_size = args.micro_batch_size
    in_channels = 4
    video_lenth = 16

    if not os.path.exists(traced_path):
        vae_model = VideoAutoencoderKL(from_pretrained=model_path, micro_batch_size=micro_batch_size)
        dummy_input = (
            torch.ones([batch_size, in_channels, video_lenth, height, width], dtype=torch.float32)
        )
        vae = VaeExport(vae_model)
        vae.eval()
        torch.jit.trace(vae, dummy_input).save(traced_path)
    if not os.path.exists(compiled_path):
        model = torch.jit.load(traced_path).eval()
        compiled_model = mindietorch.compile(
            model,
            inputs=[mindietorch.Input((batch_size, in_channels, video_lenth, height, width),
                                    dtype=mindietorch.dtype.FLOAT)],
            allow_tensor_replace_int=True,
            require_full_compilation=True,
            truncate_long_and_double=True,
            precision_policy=_enums.PrecisionPolicy.FP16,
            soc_version="Ascend910B4",
            optimization_level=0
        )
        torch.jit.save(compiled_model, compiled_path)

def main():
    args = parse_arguments()
    device_id = args.device_id
    save_dir = args.output_dir
    mindietorch.set_device(device_id)
    batch_size = 1

    export_textencoder(args, save_dir, batch_size)
    export_dit(args, save_dir, batch_size*2)
    export_vae(args, save_dir, batch_size)
    print("export model done!")
    mindietorch.finalize()

if __name__ == "__main__":
    main()