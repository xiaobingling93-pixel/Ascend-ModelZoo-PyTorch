import os
import torch
import mindietorch
from mindietorch import _enums
import argparse
from argparse import Namespace
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel

def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save pt&ts models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./stable-audio-open-1.0",
        help="The path of the pretrained stable-audio-open-1.0.",
    )
    parser.add_argument(
        "--soc",
        help="soc_version.",
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="NPU device.",
    )
    return parser.parse_args()

def export(args) -> None:
    print("Exporting the dit...")
    audio_dit = StableAudioDiTModel.from_pretrained(args.model+"/transformer").to("cpu")
    audio_dit.to(torch.float32).eval()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, mode=0o640)
    
    #trace
    dit_pt_path = os.path.join(args.output_dir, f"dit.pt")
    if not os.path.exists(dit_pt_path):
        prompt = torch.randn([2, 64, 1024], dtype=torch.float32)
        t = torch.ones([1], dtype=torch.float32)
        encoder_hidden_states = torch.randn([2, 130, 768], dtype=torch.float32)
        global_hidden_states = torch.randn([2, 1, 1536], dtype=torch.float32)
        rotary_embedding = torch.randn([2, 1025, 32], dtype=torch.float32)
        dummy_input = (prompt, t, encoder_hidden_states, global_hidden_states, rotary_embedding)
        torch.jit.trace(audio_dit, dummy_input).save(dit_pt_path)

    #compile
    dit_compiled_path = os.path.join(args.output_dir, f"dit_compile.ts")
    if not os.path.exists(dit_compiled_path):
        dit = torch.jit.load(dit_pt_path).eval()
        compiled_dit = (
            mindietorch.compile(dit,
                                inputs=[mindietorch.Input((2, 64, 1024),
                                                           dtype=mindietorch.dtype.FLOAT16),
                                        mindietorch.Input((1,),
                                                           dtype=mindietorch.dtype.FLOAT16),
                                        mindietorch.Input((2, 130, 768),
                                                           dtype=mindietorch.dtype.FLOAT16),
                                        mindietorch.Input((2, 1, 1536),
                                                           dtype=mindietorch.dtype.FLOAT16),
                                        mindietorch.Input((2, 1025, 32),
                                                           dtype=mindietorch.dtype.FLOAT16)],
                                allow_tensor_replace_int=False,
                                require_full_compilation=False,
                                truncate_long_and_double=False,
                                soc_version=args.soc,
                                precision_policy=_enums.PrecisionPolicy.FP16,
                                optimization_level=0
                                ))
        torch.jit.save(compiled_dit, dit_compiled_path)

def main(args):
    mindietorch.set_device(args.device)
    export(args)
    print("Done.")
    mindietorch.finalize()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)