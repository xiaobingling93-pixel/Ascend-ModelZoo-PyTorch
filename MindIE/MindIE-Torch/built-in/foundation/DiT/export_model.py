import argparse
import os
from argparse import Namespace
import torch
from models import DiT_models
from download import find_model
from diffusers.models import AutoencoderKL
import mindietorch
from mindietorch import _enums

class DiTExport(torch.nn.Module):
    def __init__(self, dit_model):
        super().__init__()
        self.dit_model = dit_model

    def forward(self, x, t, y):
        return self.dit_model(x, t, y)
    
class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        return self.vae_model.decode(latents)[0]

def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save models"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./DiT-XL-2-256x256.pt",
        help="Path or name of the pre-trained model."
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="./sd-vae-ft-ema",
        help="Path or name of the vae pre-trained model."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--parallel", action="store_true", help="Use parallel during inference")
    parser.add_argument(
        "--soc",
        type=str,
        default="Duo",
        choices=["Duo", "A2"],
        help="soc_version"
    )
    return parser.parse_args()

def export_dit(args, soc_version):
    print(f"start trace dit_{args.image_size}---------->")
    dit_path = os.path.join(args.output_dir, "dit")
    if not os.path.exists(dit_path):
        os.makedirs(dit_path, mode=0o640)
    device = "cpu"
    ckpt_path = args.ckpt
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    if args.parallel:
        batch = 1
        traced_path = os.path.join(dit_path, f"dit_model_{args.image_size}_parallel.pt")
        compiled_path = os.path.join(dit_path, f"dit_model_{args.image_size}_parallel_compiled.pt")
    else:
        batch = 2
        traced_path = os.path.join(dit_path, f"dit_model_{args.image_size}.pt")
        compiled_path = os.path.join(dit_path, f"dit_model_{args.image_size}_compiled.pt")
    dummy_input = (
        torch.ones([batch, 4, latent_size, latent_size], dtype=torch.float32),
        torch.ones([batch,], dtype=torch.int64),
        torch.ones([batch,], dtype=torch.int64),
    )
    # trace模型
    if not os.path.exists(traced_path):
        dit_model = DiTExport(model)
        dit_model.eval()
        torch.jit.trace(dit_model, dummy_input).save(traced_path)

    # compile模型
    print(f"start compile dit_{args.image_size}---------->")
    inputs = [
        mindietorch.Input((batch, 4, latent_size, latent_size),
                          dtype=mindietorch.dtype.FLOAT),
        mindietorch.Input((batch,),
                          dtype=mindietorch.dtype.INT64),
        mindietorch.Input((batch,),
                          dtype=mindietorch.dtype.INT64)                
    ]
    if not os.path.exists(compiled_path):
        jit_model = torch.jit.load(traced_path).eval()
        compiled_model = (
            mindietorch.compile(jit_model,
                                inputs=inputs,
                                allow_tensor_replace_int=True,
                                require_full_compilation=True,
                                truncate_long_and_double=True,
                                soc_version=soc_version,
                                precision_policy=_enums.PrecisionPolicy.FP16,
                                optimization_level=0)
        )
        torch.jit.save(compiled_model, compiled_path)

def export_vae(args, soc_version):
    kind = args.vae_model[-3:]
    print(f"start trace vae_{kind}_{args.image_size}---------->")
    vae_path = os.path.join(args.output_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    device = "cpu"
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    latent_size = args.image_size // 8
    batch = 1
    dummy_input = (
        torch.ones([batch, 4, latent_size, latent_size], dtype=torch.float32)
    )
    traced_path = os.path.join(vae_path, f"vae_{kind}_{args.image_size}.pt")
    compiled_path = os.path.join(vae_path, f"vae_{kind}_{args.image_size}_compiled.pt")

    # trace模型
    if not os.path.exists(traced_path):
        vae_model = VaeExport(vae)
        vae_model.eval()
        torch.jit.trace(vae_model, dummy_input).save(traced_path)
    # compile模型
    print(f"start compile vae_{kind}_{args.image_size}---------->")
    inputs = [
        mindietorch.Input((batch, 4, latent_size, latent_size),
                          dtype=mindietorch.dtype.FLOAT)               
    ]
    if not os.path.exists(compiled_path):
        jit_model = torch.jit.load(traced_path).eval()
        compiled_model = (
            mindietorch.compile(jit_model,
                                inputs=inputs,
                                allow_tensor_replace_int=True,
                                require_full_compilation=True,
                                truncate_long_and_double=True,
                                soc_version=soc_version,
                                precision_policy=_enums.PrecisionPolicy.FP16,
                                optimization_level=0)
        )
        torch.jit.save(compiled_model, compiled_path)
    

def main():
    args = parse_arguments()
    device_id = args.device
    mindietorch.set_device(device_id)

    if args.soc == "Duo":
        soc_version = "Ascend310P3"
    elif args.soc == "A2":
        soc_version = "Ascend910B4"
    else:
        print("Unsupport soc_version")
        return
    export_dit(args, soc_version)
    export_vae(args, soc_version)
    mindietorch.finalize()
    print("Done")

if __name__ == "__main__":
    main()
