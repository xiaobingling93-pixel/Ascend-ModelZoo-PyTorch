import torch
import torch_npu
import sys
import time
import json
import os
import argparse
import soundfile as sf
from safetensors.torch import load_file
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
from diffusers import StableAudioPipeline
from transformers import T5TokenizerFast
from transformers import T5EncoderModel
from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel
from diffusers.schedulers.scheduling_cosine_dpmsolver_multistep import CosineDPMSolverMultistepScheduler

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="Berlin techno, rave, drum machine, kick, ARP synthesizer, dark, moody, hypnotic, evolving, 135 BPM. Loop.",
        help="The prompt or prompts to guide audio generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="The prompt or prompts to guide what to not include in audio generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality audio at the expense of slower inference.",
    )
    parser.add_argument(
        "--latents",
        type=torch.Tensor,
        default=torch.randn(1, 64, 1024,dtype=torch.float16),
        help="Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for audio generation.",
    )
    parser.add_argument(
        "--stable_audio_open_dir",
        type=str,
        default="./stable-audio-open-1.0",
        help="The path of stable-audio-open-1.0.",
    )
    parser.add_argument(
        "--audio_start_in_s",
        type=float,
        default=0,
        help="Audio start index in seconds.",
    )
    parser.add_argument(
        "--audio_end_in_s",
        type=float,
        default=10,
        help="Audio end index in seconds.",
    )
    parser.add_argument(
        "--num_waveforms_per_prompt",
        type=int,
        default=1,
        help="The number of waveforms to generate per prompt.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7,
        help="A higher guidance scale value encourages the model to generate audio that is closely linked to the text `prompt` at the expense of lower sound quality.",
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

    with open(args.stable_audio_open_dir + "/vae/config.json", "r", encoding="utf-8") as reader:
        data = reader.read()
    json_data = json.loads(data)
    init_dict = {key: json_data[key] for key in json_data}
    print(init_dict)
    vae = AutoencoderOobleck(**init_dict)
    vae.load_state_dict(load_file(args.stable_audio_open_dir + "/vae/diffusion_pytorch_model.safetensors"), strict=False)

    tokenizer = T5TokenizerFast.from_pretrained(args.stable_audio_open_dir + "/tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.stable_audio_open_dir + "/text_encoder")
    projection_model = StableAudioProjectionModel.from_pretrained(args.stable_audio_open_dir + "/projection_model")
    audio_dit = StableAudioDiTModel.from_pretrained(args.stable_audio_open_dir + "/transformer")
    scheduler = CosineDPMSolverMultistepScheduler.from_pretrained(args.stable_audio_open_dir + "/scheduler")

    vae = vae.to("npu").to(torch.float16).eval()
    text_encoder = text_encoder.to("npu").to(torch.float16).eval()
    projection_model = projection_model.to("npu").to(torch.float16).eval()
    audio_dit = audio_dit.to("npu").to(torch.float16).eval()

    pipe = StableAudioPipeline(vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
        projection_model=projection_model, transformer=audio_dit, scheduler=scheduler)
    pipe.to("npu")

    with torch.no_grad():
        audio = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            latents=args.latents.to("npu"),
            audio_end_in_s=args.audio_end_in_s,
            num_waveforms_per_prompt=args.num_waveforms_per_prompt,
        ).audios

    output = audio[0].T.float().cpu().numpy()
    sf.write(args.save_dir+"/audio.wav", output, pipe.vae.sampling_rate)


if __name__ == "__main__":
    main()