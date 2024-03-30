# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import time
import json
import argparse

import aclruntime
from ais_bench.infer.interface import InferSession
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler

from pipeline_ascend_stable_diffusionxl import AscendStableDiffusionXLPipeline


class PromptLoader:
    def __init__(
        self,
        prompt_file: str,
        prompt_file_type: str,
        batch_size: int,
        num_images_per_prompt: int=1,
        max_num_prompts: int=0
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)

        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration

        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret

    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))
                
    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="A prompt file used to generate images.",
    )
    parser.add_argument(
        "--prompt_file_type", 
        choices=["plain", "parti"],
        default="plain", 
        help="Type of prompt file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Base path of om models.",
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./results", 
        help="Path to save result images.",
    )
    parser.add_argument(
        "--info_file_save_path", 
        type=str, 
        default="./image_info.json", 
        help="Path to save image information file.",
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50, 
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="Number of images generated for each prompt.",
    )
    parser.add_argument(
        "--max_num_prompts",
        default=0,
        type=int,
        help="Limit the number of prompts (0: no limit).",
    )
    parser.add_argument(
        "--scheduler", 
        choices=["DDIM", "Euler", "DPM"],
        default="DDIM", 
        help="Type of Sampling methods. Can choose from DDIM, Euler, DPM",
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=0, 
        help="NPU device id."
    )
    parser.add_argument(
        "-bs",
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size."
    )
    parser.add_argument(
        "--use_cache", 
        action="store_true",
        help="Use cache during inference."
    )
    parser.add_argument(
        "--cache_steps", 
        type=str, 
        default="1,2,4,6,7,9,10,12,13,14,16,18,19,21,23,24,26,27,29,\
                30,31,33,34,36,37,39,40,42,43,45,47,48,49", 
        help="Steps to use cache data."
    )


    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    device = args.device

    pipe = AscendStableDiffusionXLPipeline.from_pretrained(args.model).to("cpu")

    if args.scheduler == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "DPM":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


    encoder_om = os.path.join(args.model_dir, "text_encoder", "text_encoder.om")
    encoder_om_2 = os.path.join(args.model_dir, "text_encoder", "text_encoder_2.om")
    vae_om = os.path.join(args.model_dir, "vae", "vae.om")
    scheduler_om = os.path.join(args.model_dir, "ddim", "ddim.om")

    encoder_session = InferSession(device, encoder_om)
    encoder_session_2 = InferSession(device, encoder_om_2)
    vae_session = InferSession(device, vae_om)
    scheduler_session = InferSession(device, scheduler_om)

    skip_status = [0] * args.steps
    if args.use_cache:
        for i in args.cache_steps.split(','):
            if int(i) >= args.steps:
                continue
            skip_status[int(i)] = 1
        unet_cache_om = os.path.join(args.model_dir, "unet", "unet_cache.om")
        unet_skip_om = os.path.join(args.model_dir, "unet", "unet_skip.om")
        unet_session = [
            aclruntime.InferenceSession(unet_cache_om, device, aclruntime.session_options()),
            aclruntime.InferenceSession(unet_skip_om, device, aclruntime.session_options()),
        ]
    else:
        unet_cache_om = os.path.join(args.model_dir, "unet", "unet.om")
        unet_skip_om = ""
        unet_session = [
            aclruntime.InferenceSession(unet_cache_om, device, aclruntime.session_options()),
            None,
        ]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o744)

    use_time = 0

    prompt_loader = PromptLoader(args.prompt_file, 
                                 args.prompt_file_type, 
                                 args.batch_size,
                                 args.num_images_per_prompt,
                                 args.max_num_prompts)

    prompts_2 = ""
    infer_num = 0
    image_info = []
    current_prompt = None
    for _, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']
        
        print(f"[{infer_num + n_prompts}/{len(prompt_loader)}]: {prompts}")
        infer_num += args.batch_size

        start_time = time.time()
        images = pipe.ascend_infer(
            prompts,
            prompts_2,
            encoder_session,
            encoder_session_2,
            unet_session,
            scheduler_session,
            vae_session,
            skip_status,
            device_id=device,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
        )

        use_time += time.time() - start_time

        for j in range(n_prompts):
            image_save_path = os.path.join(save_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)
        
    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR|os.O_CREAT, 0o644), "w") as f:
        json.dump(image_info, f)

    print(
        f"[info] infer number: {infer_num}; use time: {use_time:.3f}s; "
        f"average time: {use_time/infer_num:.3f}s"
    )

    # free npu resource
    encoder_session.free_resource()
    encoder_session_2.free_resource()
    vae_session.free_resource()
    scheduler_session.free_resource()
    unet_session[0].free_resource()
    if args.use_cache:
        unet_session[1].free_resource()
    InferSession.finalize()


if __name__ == "__main__":
    main()
