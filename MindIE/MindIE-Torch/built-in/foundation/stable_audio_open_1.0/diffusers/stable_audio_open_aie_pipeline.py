import torch
import torch_npu
import mindietorch
import sys
import time
import json
import os
import argparse
import copy
import soundfile as sf
from safetensors.torch import load_file
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
from diffusers import StableAudioPipeline
from transformers import T5TokenizerFast
from transformers import T5EncoderModel
from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel
from diffusers.schedulers.scheduling_cosine_dpmsolver_multistep import CosineDPMSolverMultistepScheduler
from typing import Optional

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts.txt",
        help="The prompts file to guide audio generation.",
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
        "--model",
        type=str,
        default="./stable-audio-open-1.0",
        help="The path of stable-audio-open-1.0.",
    )
    parser.add_argument(
        "--audio_end_in_s",
        nargs='+',
        default=[10],
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
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save pt&ts models.",
    )
    return parser.parse_args()

class AieDiTModel():
    def __init__(
        self,
        config:None,
        args:None,
    ):
        super().__init__()
        self.config = config
        dit_compiled_path = os.path.join(args.output_dir+"/dit_compile.ts")
        if os.path.exists(dit_compiled_path):
            self.compiled_dit = torch.jit.load(dit_compiled_path).eval()
        else:
            print("%s have no dit_compile.ts, please run export_ts.py first, program is exiting..."%(args.output_dir))
            sys.exit()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        return_dict: Optional[bool] = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ):
        rotary_embedding = torch.stack([torch.tensor(re,dtype=torch.float16) for re in rotary_embedding]).to("npu")
        timestep = torch.tensor(timestep, dtype=torch.float16)
        output = self.compiled_dit(hidden_states, timestep, encoder_hidden_states, global_hidden_states,rotary_embedding).to(torch.float16)
        return output

def main():
    args = parse_arguments()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch_npu.npu.set_device(args.device)
    torch.manual_seed(1)
    latents = torch.randn(1, 64, 1024,dtype=torch.float16,device="cpu").to("npu")
    with open(args.model + "/vae/config.json", "r", encoding="utf-8") as reader:
        data = reader.read()
    json_data = json.loads(data)
    init_dict = {key: json_data[key] for key in json_data}
    vae = AutoencoderOobleck(**init_dict)
    vae.load_state_dict(load_file(args.model + "/vae/diffusion_pytorch_model.safetensors"), strict=False)

    tokenizer = T5TokenizerFast.from_pretrained(args.model + "/tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.model + "/text_encoder")
    projection_model = StableAudioProjectionModel.from_pretrained(args.model + "/projection_model")
    audio_dit0 = StableAudioDiTModel.from_pretrained(args.model + "/transformer")
    config = copy.deepcopy(audio_dit0.config)
    del audio_dit0
    audio_dit = AieDiTModel(config,args)
    scheduler = CosineDPMSolverMultistepScheduler.from_pretrained(args.model + "/scheduler")

    npu_stream = torch_npu.npu.Stream()
    vae = vae.to("npu").to(torch.float16).eval()
    text_encoder = text_encoder.to("npu").to(torch.float16).eval()
    projection_model = projection_model.to("npu").to(torch.float16).eval()

    pipe = StableAudioPipeline(vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
        projection_model=projection_model, transformer=audio_dit, scheduler=scheduler)

    total_time = 0
    prompts_num = 0
    average_time = 0
    skip = 2
    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        for i, prompt in enumerate(f):
            with torch.no_grad():
                npu_stream.synchronize()
                audio_end_in_s = float(args.audio_end_in_s[i]) if (len(args.audio_end_in_s) > i) else 10.0
                begin = time.time()
                audio = pipe(
                    prompt=prompt.strip(),
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    latents=latents,
                    audio_end_in_s=audio_end_in_s,
                    num_waveforms_per_prompt=args.num_waveforms_per_prompt,
                ).audios
                npu_stream.synchronize()
                end = time.time()
                if i > skip-1:
                    total_time += end - begin
            prompts_num = i+1
            output = audio[0].T.float().cpu().numpy()
            sf.write(args.save_dir+"/audio_by_prompt"+str(prompts_num)+".wav", output, pipe.vae.sampling_rate)
    if prompts_num>skip:
        average_time = total_time/(prompts_num-skip)
    else:
        print("Infer average time skip first two prompts, make sure prompts.txt has three more prompts")
    print(f"Infer average time: {average_time:.3f}s\n")
    mindietorch.finalize()

if __name__ == "__main__":
    main()