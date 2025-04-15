import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union

from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist
from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from hyvideo.vae import load_vae
from hyvideo.modules import load_model
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.data_utils import align_to
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.modules.fp8_optimization import convert_fp8_linear
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from hyvideo.utils.parallel_mgr import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    initialize_model_parallel,
    init_distributed_environment,
    all_gather
)
from hyvideo.vae.parallel_layers import (
    PatchCausalConv3d,  
    PatchConv3d, 
    PatchGroupNorm3d, 
    BaseModule,
    AttnProcessor2_0_fa,
    patchify, 
    depatchify,
    register_upsample_forward,
    register_vae_midblock_forward,
)
from hyvideo.vae.unet_causal_3d_blocks import CausalConv3d, UNetMidBlockCausal3D, UpsampleCausal3D
from hyvideo.vae.vae import DecoderOutput
from hyvideo.vae.vae_parallel import parallel_vae_tile
from hyvideo.modules.new_parallel import split_sequence, gather_sequence
ATTN_PARALLEL = False


def parallel_full_model_warp(vae, dim=-1):
    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()

    decoder = vae.decoder
    post_quant_conv = vae.post_quant_conv
    vae.post_quant_conv = PatchConv3d(post_quant_conv, split_dim=dim)

    
    for name, module in decoder.named_modules():
        if isinstance(module, BaseModule):
            continue
        for subname, submodule in module.named_children():
            if isinstance(submodule, CausalConv3d):
                wrapped_submodule = PatchCausalConv3d(submodule, split_dim=dim, num_blocks=2)
                setattr(module, subname, wrapped_submodule)

            elif isinstance(submodule, torch.nn.GroupNorm):
                wrapped_submodule = PatchGroupNorm3d(submodule, split_dim=dim)
                setattr(module, subname, wrapped_submodule)
            elif subname == "attentions":
                submodule[0].processor = AttnProcessor2_0_fa(world_size, rank, split_dim=dim)
                setattr(module, subname, submodule)
                if isinstance(module, UNetMidBlockCausal3D):
                    register_vae_midblock_forward(module)
            elif isinstance(submodule, UpsampleCausal3D):
                register_upsample_forward(submodule)


def parallelize_vae(pipe):
    vae = pipe.vae
    parallel_dim = -1
    parallel_full_model_warp(vae, parallel_dim)
    
    @functools.wraps(vae.__class__._decode)
    def new_decode(
        self, 
        z: torch.FloatTensor, 
        return_dict: bool = True
        ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images/videos using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """

        parallel_dim = -1
        parallel_overlap = True
        world_size = get_sequence_parallel_world_size()
        rank = get_sequence_parallel_rank()

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.

        z_patch = patchify(z, parallel_dim, parallel_overlap, world_size, rank)
        z_patch = self.post_quant_conv(z_patch)
        dec_patch = self.decoder(z_patch)
        decoded_full = depatchify(dec_patch, parallel_dim, parallel_overlap, world_size, rank)

        if not return_dict:
            return (decoded_full,)

        return DecoderOutput(sample=decoded_full)
    
    new_decode = new_decode.__get__(vae)
    vae._decode = new_decode
    pipe.vae = vae


def parallelize_vae_tiling(pipe):
    vae = pipe.vae
    parallel_dim = -1
    parallel_full_model_warp(vae, parallel_dim)
    
    @functools.wraps(vae.__class__._decode)
    def new_temporal_tiled_decode(
        self, 
        z: torch.FloatTensor, 
        return_dict: bool = True
        ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images/videos using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """

        parallel_dim = -1
        parallel_overlap = True
        world_size = get_sequence_parallel_world_size()
        rank = get_sequence_parallel_rank()

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i: i + self.tile_latent_min_tsize + 1, :, :]
            tile_patch = patchify(tile, parallel_dim, parallel_overlap, world_size, rank)
            tile_patch = self.post_quant_conv(tile_patch)
            decoded_patch = self.decoder(tile_patch)
            decoded = depatchify(decoded_patch, parallel_dim, parallel_overlap, world_size, rank)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit + 1, :, :])

        dec = torch.cat(result_row, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    
    new_temporal_tiled_decode = new_temporal_tiled_decode.__get__(vae)
    vae.temporal_tiled_decode = new_temporal_tiled_decode
    pipe.vae = vae


def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.seq_forward

    @functools.wraps(transformer.__class__.seq_forward)
    def new_seq_forward(
        self,
        img,
        txt,
        text_mask,
        freqs_cos,
        freqs_sin,
        vec,
        t_idx
    ):
        seqlen = img.shape[1]
        divisable = True
        if seqlen % get_sequence_parallel_world_size() != 0:
            divisable = False
        img = split_sequence(img)
        freqs_cos = split_sequence(freqs_cos, dim=0)
        freqs_sin = split_sequence(freqs_sin, dim=0)
        
        from hyvideo.modules.attn_layer import xFuserLongContextAttention
        
        for block in transformer.double_blocks + transformer.single_blocks:
            block.hybrid_seq_parallel_attn = xFuserLongContextAttention(divisable=divisable)

        output = original_forward(
            img,
            txt,
            text_mask,
            freqs_cos,
            freqs_sin,
            vec,
            t_idx
        )

        output = gather_sequence(output)

        return output

    new_seq_forward = new_seq_forward.__get__(transformer)
    transformer.seq_forward = new_seq_forward
    

class Inference(object):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload

        self.args = args
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = logger
        self.parallel_args = parallel_args

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is 0.
        """
        # ========================================================================        
        # ==================== Initialize Distributed Environment ================
        if args.ulysses_degree > 1 or args.ring_degree > 1:

            dist.init_process_group("hccl")

            if dist.get_world_size() != args.ring_degree * args.ulysses_degree:
                raise ValueError(f"number of NPUs should be equal to ring_degree * ulysses_degree.")
            
            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size(), backend="hccl")
            
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=args.ring_degree,
                ulysses_degree=args.ulysses_degree,
                backend="hccl"
            )
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        else:
            if device is None:
                device = f"npu:{args.device_id}"
                torch.npu.set_device(device)
        parallel_args = {"ulysses_degree": args.ulysses_degree, "ring_degree": args.ring_degree}

        # ======================== Get the args path =============================

        # Disable gradient
        torch.set_grad_enabled(False)

        # =========================== Build main model ===========================
        logger.info("Building model...")
        factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
        in_channels = args.latent_channels
        out_channels = args.latent_channels

        model = load_model(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )

        model = model.to(device)
        model = Inference.load_state_dict(args, model, pretrained_model_path)
        model.eval()

        # ============================= Build extra models ========================
        # VAE
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae,
            args.vae_precision,
            logger=logger,
            device=device,
            vae_path=args.vae_path
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        # Text encoder
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get(
                "crop_start", 0
            )
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # prompt_template
        prompt_template = (
            PROMPT_TEMPLATE[args.prompt_template]
            if args.prompt_template is not None
            else None
        )

        # prompt_template_video
        prompt_template_video = (
            PROMPT_TEMPLATE[args.prompt_template_video]
            if args.prompt_template_video is not None
            else None
        )

        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=device,
            text_encoder_path=args.text_encoder_path
        )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=args.text_encoder_2,
                max_length=args.text_len_2,
                text_encoder_precision=args.text_encoder_precision_2,
                tokenizer_type=args.tokenizer_2,
                reproduce=args.reproduce,
                logger=logger,
                device=device,
                text_encoder_path=args.text_encoder_2_path
            )

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            use_cpu_offload=args.use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args
        )

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path):
        load_key = args.load_key
        dit_weight = Path(args.dit_weight)

        if dit_weight is None:
            model_dir = pretrained_model_path / f"t2v_{args.model_resolution}"
            files = list(model_dir.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {model_dir}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(
                        f"Multiple model weights found in {dit_weight}, using {model_path}"
                    )
                bare_model = False
            else:
                raise ValueError(
                    f"Invalid model path: {dit_weight} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        else:
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith("pytorch_model_"):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith("_model_states.pt") for f in files):
                    files = [f for f in files if str(f).endswith("_model_states.pt")]
                    model_path = files[0]
                    if len(files) > 1:
                        logger.warning(
                            f"Multiple model weights found in {dit_weight}, using {model_path}"
                        )
                    bare_model = False
                else:
                    raise ValueError(
                        f"Invalid model path: {dit_weight} with unrecognized weight format: "
                        f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                        f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                        f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                        f"specific weight file, please provide the full path to the file."
                    )
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = "unknown"
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")

        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}."
                )
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size


class HunyuanVideoSampler(Inference):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=0,
        logger=None,
        parallel_args=None
    ):
        super().__init__(
            args,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args
        )

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )

        self.default_negative_prompt = NEGATIVE_PROMPT
        if self.parallel_args['ulysses_degree'] > 1 or self.parallel_args['ring_degree'] > 1:
            parallelize_transformer(self.pipeline)
            if args.vae_parallel:
                if get_sequence_parallel_world_size() in [8, 16]:
                    if get_sequence_parallel_world_size() > 8:
                        parallel_vae_tile(self.pipeline.vae, "decode", "decoder.forward")
                    else:
                        parallelize_vae_tiling(self.pipeline)
            import deepspeed
            self.pipeline.text_encoder.model = deepspeed.init_inference(
                self.pipeline.text_encoder.model,
                tensor_parallel={"tp_size": get_sequence_parallel_world_size()}
            )

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
        data_type="video",
    ):
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload(device=f"npu:{args.device_id}")
        else:
            pipeline = pipeline.to(device)

        return pipeline

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model.patch_size, int):
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            rope_sizes = [
                s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.args.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                infer_steps (int): The number of inference steps. Default is 100.
        """
        out_dict = dict()

        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [torch.Generator(torch.device('cpu')).manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver
        )
        self.pipeline.scheduler = scheduler

        # ========================================================================
        # Build Rope freqs
        # ========================================================================
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )
        freqs_cos = freqs_cos.to(self.device)
        freqs_sin = freqs_sin.to(self.device)
        n_tokens = freqs_cos.shape[0]

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        torch.npu.synchronize()
        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt
        
        torch.npu.synchronize()
        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict