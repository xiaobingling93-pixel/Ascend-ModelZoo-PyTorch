import torch
import torch_npu
import time
import os
import argparse
from safetensors.torch import load_file
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts.txt",
        help="The prompts file to guide audio generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./stable-audio-open-1.0",
        help="The path of stable-audio-open-1.0.",
    )
    parser.add_argument(
        "--seconds_total",
        nargs='+',
        default=[10],
        help="Audio end index in seconds.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device id.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result audio files.",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch_npu.npu.set_device(args.device)
    npu_stream = torch_npu.npu.Stream()
    
    model, model_config = get_pretrained_model(args.model)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to("npu").to(torch.float16).eval()

    conditioning = [{
        "prompt":"",
        "seconds_start": 0,
        "seconds_total": 0,
    }]
    total_time = 0
    prompts_num = 0
    average_time = 0
    skip = 2
    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        for i, prompt in enumerate(f):
            with torch.no_grad():
                conditioning[0]["prompt"] = prompt
                conditioning[0]["seconds_total"] = float(args.seconds_total[i]) if (len(args.seconds_total) > i) else 10.0

                npu_stream.synchronize()
                begin = time.time()
                output = generate_diffusion_cond(
                    model,
                    steps=args.num_inference_steps,
                    cfg_scale=7,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device="npu"
                )
                npu_stream.synchronize()
                end = time.time()
                if i > skip-1:
                    total_time += end - begin
            prompts_num = i+1
            waveform_start = int(conditioning[0]["seconds_start"] * sample_rate)
            waveform_end = int(conditioning[0]["seconds_total"] * sample_rate)
            output = output[:, :, waveform_start:waveform_end]
            output = rearrange(output, "b d n -> d (b n)")
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1,1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(args.save_dir + "/audio_by_prompt" + str(prompts_num) + ".wav", output, sample_rate)
    if prompts_num > skip:
        average_time = total_time / (prompts_num-skip)
    else:
        print("Infer average time skip first two prompts, make sure prompts.txt has three more prompts")
    print(f"Infer average time: {average_time:.3f}s\n")

if __name__ == "__main__":
    main()