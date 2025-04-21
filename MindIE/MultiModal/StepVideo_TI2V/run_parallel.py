from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
import torch_npu
from stepvideo.config import parse_args
from stepvideo.parallel import initialize_parall_group, get_parallel_group
from stepvideo.parallel import enable_llm_tensor_model_parallel, get_llm_tensor_model_parallel_world_size, get_llm_tensor_model_parallel_rank, get_llm_tensor_model_parallel_group
from stepvideo.utils import setup_seed
from xfuser.model_executor.models.customized.step_video_t2v.tp_applicator import TensorParallelApplicator
from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from api.call_remote_server import CaptionPipeline, StepVaePipeline

if __name__ == "__main__":
    torch_npu.npu.config.allow_internal_format = False
    args = parse_args()
    initialize_parall_group(ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree,
                            tensor_parallel_degree=args.tensor_parallel_degree)

    local_rank = get_parallel_group().local_rank
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)

    setup_seed(args.seed)

    pipeline = StepVideoPipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(device="cpu")

    if args.tensor_parallel_degree > 1:
        tp_applicator = TensorParallelApplicator(get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank())
        tp_applicator.apply_to_model(pipeline.transformer)
    pipeline.transformer = pipeline.transformer.to(device)

    if args.use_dit_cache:
        from mindiesd.layers.cache_mgr import CacheManager, DitCacheConfig
        config = DitCacheConfig(step_start=6, step_interval=2, block_start=11, num_blocks=31)
        cache = CacheManager(config)
        pipeline.transformer.cache = cache

    def patch_encode_prompt():
        enable_llm_tensor_model_parallel()
        caption_pipeline = CaptionPipeline(llm_dir="/home/data/stepvideo-ti2v/step_llm", clip_dir="/home/data/stepvideo-ti2v/hunyuan_clip", device='cpu')
        if args.tensor_parallel_degree > 1:
            llm_tp_applicator = TensorParallelApplicator(get_llm_tensor_model_parallel_world_size(), get_llm_tensor_model_parallel_rank(), tp_group=get_llm_tensor_model_parallel_group())
            llm_tp_applicator.apply_to_llm_model(caption_pipeline.text_encoder)
            llm_tp_applicator.apply_to_model(caption_pipeline.clip)

        caption_pipeline.text_encoder = caption_pipeline.text_encoder.to(device)
        caption_pipeline.clip = caption_pipeline.clip.to(device)

        def encode_prompt(
            prompt: str,
            neg_magic: str = '',
            pos_magic: str = '',
        ):

            prompts = [prompt + pos_magic]
            bs = len(prompts)
            prompts += [neg_magic] * bs

            data = caption_pipeline.embedding(prompts)
            prompt_embeds, prompt_attention_mask, clip_embedding = data['y'].to(device), data['y_mask'].to(device), data['clip_embedding'].to(device)
            return prompt_embeds, clip_embedding, prompt_attention_mask

        pipeline.encode_prompt = encode_prompt

    def patch_vae():
        vae_pipeline = StepVaePipeline(vae_dir="/home/data/stepvideo-ti2v/vae", device='cpu')
        vae_pipeline.vae = vae_pipeline.vae.to(device)

        def encode_vae(img):
            latents = vae_pipeline.encode(img)
            return latents
        
        def decode_vae(samples):
            samples = vae_pipeline.decode(samples)
            return samples
        pipeline.encode_vae = encode_vae
        pipeline.decode_vae = decode_vae

    patch_encode_prompt()
    patch_vae()

    prompt = args.prompt

# warm up
    videos = pipeline(
        prompt=prompt,
        first_image=args.first_image_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=2,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=args.output_file_name or prompt[:50],
        motion_score=args.motion_score,
    )

    import time
    torch.npu.synchronize()
    start_time = time.time()
    videos = pipeline(
        prompt=prompt,
        first_image=args.first_image_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=args.output_file_name or prompt[:50],
        motion_score=args.motion_score,
    )
    torch.npu.synchronize()
    print(f"E2E time: {time.time() - start_time}s")

    dist.destroy_process_group()