# Copyright 2024 Huawei Technologies Co., Ltd
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
import logging
import os
import argparse
from argparse import Namespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import StableDiffusion3Pipeline
import mindietorch
from compile_model import *


def check_owner(path: str):
    """
    check the path owner
    param: the input path
    return: whether the path owner is current user or not
    """
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    user_check = path_owner == os.getuid() and path_owner == os.geteuid()
    return path_owner == 0 or path_gid in os.getgroups() or user_check


def path_check(path: str):
    """
    check path
    param: path
    return: data real path after check
    """
    if os.path.islink(path) or path is None:
        raise RuntimeError("The path should not be None or a symbolic link file.")
    path = os.path.realpath(path)
    if not check_owner(path):
        raise RuntimeError("The path is not owned by current user or root.")
    return path


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save pt models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./stable-diffusion-3-medium-diffusers",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("-steps", "--steps", type=int, default=28, help="steps.")
    parser.add_argument("-guid", "--guidance_scale", type=float, default=7.0, help="guidance_scale")
    parser.add_argument("--use_cache", action="store_true", help="Use cache during inference.")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="Export the unet of bs=1 for parallel inferencing.")
    parser.add_argument("--soc", help="soc_version.")
    parser.add_argument("--device_type", choices=["A2", "Duo"], default="A2", help="device type.")
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="NPU device",
    )
    parser.add_argument(
        "--height",
        default=1024,
        type=int,
        help="image height",
    )
    parser.add_argument(
        "--width",
        default=1024,
        type=int,
        help="image width"
    )
    parser.add_argument(
        "--cache_param",
        type=str,
        default="1,2,20,10",
        help="Steps to use cache data."
    )
    return parser.parse_args()


def trace_clip(sd_pipeline, batch_size, clip_pt_path, clip2_pt_path, t5_pt_path):
    encoder_model = sd_pipeline.text_encoder
    encoder_2_model = sd_pipeline.text_encoder_2
    t5_model = sd_pipeline.text_encoder_3
    max_position_embeddings = encoder_model.config.max_position_embeddings
    dummy_input = torch.ones([batch_size, max_position_embeddings], dtype=torch.int64)

    if not os.path.exists(clip_pt_path):
        clip_export = ClipExport(encoder_model)
        torch.jit.trace(clip_export, dummy_input).save(clip_pt_path)
    else:
        logging.info("clip_pt_path already exists.")

    if not os.path.exists(clip2_pt_path):
        clip2_export = ClipExport(encoder_2_model)
        torch.jit.trace(clip2_export, dummy_input).save(clip2_pt_path)
    else:
        logging.info("clip2_pt_path already exists.")

    if not os.path.exists(t5_pt_path):
        t5_export = ClipExport(t5_model)
        torch.jit.trace(t5_export, dummy_input).save(t5_pt_path)
    else:
        logging.info("t5_pt_path already exists.")


def export_clip(sd_pipeline, args):
    print("Exporting the text encoder...")
    standard_path = path_check(args.output_dir)
    clip_path = os.path.join(standard_path, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o640)
    batch_size = args.batch_size
    clip_pt_path = os.path.join(clip_path, f"clip_bs{batch_size}.pt")
    clip2_pt_path = os.path.join(clip_path, f"clip2_bs{batch_size}.pt")
    t5_pt_path = os.path.join(clip_path, f"t5_bs{batch_size}.pt")
    clip1_compiled_path = os.path.join(clip_path,
                                       f"clip_bs{batch_size}_compile_{args.height}x{args.width}.ts")
    clip2_compiled_path = os.path.join(clip_path,
                                       f"clip2_bs{batch_size}_compile_{args.height}x{args.width}.ts")
    t5_compiled_path = os.path.join(clip_path,
                                    f"t5_bs{batch_size}_compile_{args.height}x{args.width}.ts")

    encoder_model = sd_pipeline.text_encoder
    max_position_embeddings = encoder_model.config.max_position_embeddings

    # trace
    trace_clip(sd_pipeline, batch_size, clip_pt_path, clip2_pt_path, t5_pt_path)

    # compile
    if not os.path.exists(clip1_compiled_path):
        model = torch.jit.load(clip_pt_path).eval()
        inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
        compile_clip(model, inputs, clip1_compiled_path, args.soc)
    else:
        logging.info("clip1_compiled_path already exists.")
    if not os.path.exists(clip2_compiled_path):
        model = torch.jit.load(clip2_pt_path).eval()
        inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
        compile_clip(model, inputs, clip2_compiled_path, args.soc)
    else:
        logging.info("clip2_compiled_path already exists.")
    if not os.path.exists(t5_compiled_path):
        model = torch.jit.load(t5_pt_path).eval()
        inputs = [mindietorch.Input((batch_size, max_position_embeddings), dtype=mindietorch.dtype.INT64)]
        compile_clip(model, inputs, t5_compiled_path, args.soc)
    else:
        logging.info("t5_compiled_path already exists.")


def export_dit(sd_pipeline, args):
    print("Exporting the dit...")
    standard_path = path_check(args.output_dir)
    dit_path = os.path.join(standard_path, "dit")
    if not os.path.exists(dit_path):
        os.makedirs(dit_path, mode=0o640)

    dit_model = sd_pipeline.transformer
    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2

    if not args.parallel:
        batch_size = args.batch_size * 2
    else:
        batch_size = args.batch_size
    sample_size = dit_model.config.sample_size
    in_channels = dit_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings * 2

    dit_pt_path = os.path.join(dit_path, f"dit_bs{batch_size}.pt")
    dit_compiled_path = os.path.join(dit_path,
                                     f"dit_bs{batch_size}_compile_{args.height}x{args.width}.ts")

    # trace
    if not os.path.exists(dit_pt_path):
        dummy_input = (
            torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
            torch.ones(
                [batch_size, max_position_embeddings, encoder_hidden_size * 2], dtype=torch.float32
            ),
            torch.ones([batch_size, encoder_hidden_size], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64)
        )
        dit = DiTExport(dit_model).eval()
        torch.jit.trace(dit, dummy_input).save(dit_pt_path)
    else:
        logging.info("dit_pt_path already exists.")

    # compile
    if not os.path.exists(dit_compiled_path):
        model = torch.jit.load(dit_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size * 2),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, encoder_hidden_size),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)]
        compile_dit(model, inputs, dit_compiled_path, args.soc)
    else:
        logging.info("dit_compiled_path already exists.")


def export_vae(sd_pipeline, args):
    print("Exporting the image decoder...")
    standard_path = path_check(args.output_dir)
    vae_path = os.path.join(standard_path, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)
    batch_size = args.batch_size
    vae_pt_path = os.path.join(vae_path, f"vae_bs{batch_size}.pt")
    vae_compiled_path = os.path.join(vae_path,
                                     f"vae_bs{batch_size}_compile_{args.height}x{args.width}.ts")

    vae_model = sd_pipeline.vae
    dit_model = sd_pipeline.transformer
    scaling_factor = vae_model.config.scaling_factor
    shift_factor = vae_model.config.shift_factor
    in_channels = vae_model.config.latent_channels
    sample_size = dit_model.config.sample_size

    # trace
    if not os.path.exists(vae_pt_path):
        dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32)
        vae_export = VaeExport(vae_model, scaling_factor, shift_factor)
        torch.jit.trace(vae_export, dummy_input).save(vae_pt_path)
    else:
        logging.info("vae_pt_path already exists.")

    # compile
    if not os.path.exists(vae_compiled_path):
        model = torch.jit.load(vae_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, sample_size, sample_size), dtype=mindietorch.dtype.FLOAT)]
        compile_vae(model, inputs, vae_compiled_path, args.soc)
    else:
        logging.info("vae_compiled_path already exists.")


def trace_scheduler(sd_pipeline, args, scheduler_pt_path):
    batch_size = args.batch_size
    if not os.path.exists(scheduler_pt_path):
        dummy_input = (
            torch.randn([batch_size, 16, 128, 128], dtype=torch.float32),
            torch.randn([batch_size, 16, 128, 128], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64)
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(sd_pipeline.scheduler.config)
        scheduler.set_timesteps(args.steps, device="cpu")

        new_scheduler = Scheduler()
        new_scheduler.eval()
        torch.jit.trace(new_scheduler, dummy_input).save(scheduler_pt_path)


def export_scheduler(sd_pipeline, args):
    print("Exporting the scheduler...")
    scheduler_path = os.path.join(args.output_dir, "scheduler")
    if not os.path.exists(scheduler_path):
        os.makedirs(scheduler_path, mode=0o744)
    batch_size = args.batch_size
    height_size, width_size = args.height // 8, args.width // 8
    scheduler_pt_path = os.path.join(scheduler_path, f"scheduler_bs{batch_size}.pt")
    scheduler_compiled_path = os.path.join(scheduler_path,
                                           f"scheduler_bs{batch_size}_compile_{args.height}x{args.width}.ts")
    in_channels = 16

    # trace
    trace_scheduler(sd_pipeline, args, scheduler_pt_path)

    # compile
    if not os.path.exists(scheduler_compiled_path):
        model = torch.jit.load(scheduler_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, in_channels, height_size, width_size),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64)
        ]
        compile_scheduler(model, inputs, scheduler_compiled_path, args.soc)


def export_dit_cache(sd_pipeline, args, if_skip, flag=""):
    print("Exporting the dit_cache...")
    cache_param = torch.zeros([4], dtype=torch.int64)
    cache_list = args.cache_param.split(',')
    cache_param[0] = int(cache_list[0])
    cache_param[1] = int(cache_list[1])
    cache_param[2] = int(cache_list[2])
    cache_param[3] = int(cache_list[3])
    dit_path = os.path.join(args.output_dir, "dit")
    if not os.path.exists(dit_path):
        os.makedirs(dit_path, mode=0o640)

    encoder_model = sd_pipeline.text_encoder
    encoder_model_2 = sd_pipeline.text_encoder_2
    dit_model = sd_pipeline.transformer
    if args.parallel or flag == "end":
        batch_size = args.batch_size
    else:
        batch_size = args.batch_size * 2
    sample_size = dit_model.config.sample_size
    in_channels = dit_model.config.in_channels
    encoder_hidden_size_2 = encoder_model_2.config.hidden_size
    encoder_hidden_size = encoder_model.config.hidden_size + encoder_hidden_size_2
    max_position_embeddings = encoder_model.config.max_position_embeddings * 2
    dit_cache_pt_path = os.path.join(dit_path, f"dit_bs{batch_size}_{if_skip}.pt")
    dit_cache_compiled_path = os.path.join(dit_path,
                                     f"dit_bs{batch_size}_{if_skip}_compile_{args.height}x{args.width}.ts")

    # trace
    if not os.path.exists(dit_cache_pt_path):
        dummy_input = (
            torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
            torch.ones([batch_size, max_position_embeddings, encoder_hidden_size * 2], dtype=torch.float32),
            torch.ones([batch_size, encoder_hidden_size], dtype=torch.float32),
            torch.ones([1], dtype=torch.int64),
            cache_param,
            torch.tensor([if_skip], dtype=torch.int64),
            torch.ones([batch_size, 4096, 1536], dtype=torch.float32),
            torch.ones([batch_size, 154, 1536], dtype=torch.float32),
        )
        print("dummy_input.shape:")
        for ele in dummy_input:
            if isinstance(ele, torch.Tensor):
                print(ele.shape)
        dit = DiTExportCache(dit_model).eval()
        torch.jit.trace(dit, dummy_input).save(dit_cache_pt_path)

    # compile
    if not os.path.exists(dit_cache_compiled_path):
        model = torch.jit.load(dit_cache_pt_path).eval()
        inputs = [
            mindietorch.Input((batch_size, in_channels, sample_size, sample_size),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, max_position_embeddings, encoder_hidden_size * 2),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, encoder_hidden_size),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input((4,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input((1,), dtype=mindietorch.dtype.INT64),
            mindietorch.Input((batch_size, 4096, 1536),
                              dtype=mindietorch.dtype.FLOAT),
            mindietorch.Input((batch_size, 154, 1536),
                              dtype=mindietorch.dtype.FLOAT),
        ]
        compile_dit_cache(model, inputs, dit_cache_compiled_path, args.soc)


def export(args) -> None:
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model).to("cpu")
    export_clip(pipeline, args)
    if args.use_cache:
        export_dit_cache(pipeline, args, 0)
        export_dit_cache(pipeline, args, 1)
        if args.device_type == "A2":
            export_dit_cache(pipeline, args, 0, "end")
            export_dit_cache(pipeline, args, 1, "end")
    else:
        export_dit(pipeline, args)
    export_vae(pipeline, args)
    export_scheduler(pipeline, args)


def main(args):
    mindietorch.set_device(args.device)
    export(args)
    print("Done.")
    mindietorch.finalize()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
