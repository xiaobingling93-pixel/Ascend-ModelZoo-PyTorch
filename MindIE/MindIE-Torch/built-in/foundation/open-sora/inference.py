import os
import torch
import time
import torch.distributed as dist
from mmengine.runner import set_random_seed
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.models.text_encoder.t5 import T5Encoder
from opensora.models.vae.vae import VideoAutoencoderKL
import mindietorch

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    image_size1, image_size2 = cfg.image_size[0], cfg.image_size[1]
    device_id = 0
    vae_npu = None
    use_mindie = False
    output_dir = cfg.output_dir
    absolute_path = os.path.abspath(output_dir)
    if cfg.use_mindie == 0:
        print("inference by CPU")
        use_mindie = False
    elif cfg.use_mindie == 1:
        print("inference by MindIE")
        use_mindie = True
        device_id = cfg.device_id
        mindietorch.set_device(device_id)
        vae_npu = torch.jit.load(f"{output_dir}/vae/vae_{image_size1}_{image_size2}_compiled.pt")

    enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    device = "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = VideoAutoencoderKL(cfg.vae_path)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = T5Encoder(
        from_pretrained=cfg.t5_path,
        model_max_length=120,
        device=device,
        use_mindie=use_mindie,
        device_id=device_id,
        absolute_path=absolute_path
    )
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=enable_sequence_parallelism,
        use_mindie=use_mindie,
        device_id=device_id,
        absolute_path=absolute_path
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    use_cache = 1
    cache_steps = []
    use_time = 0
    infer_num = 0
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        infer_num += 1
        start_time = time.time()
        samples = scheduler.sample(
            model,
            text_encoder,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
            use_cache=use_cache,
            cache_steps=cache_steps
        )
        if use_mindie:
            samples = vae_npu(samples.to(dtype).to(f"npu:{device_id}")).to('cpu')
        else:
            samples = vae.decode(samples.to(dtype))

        if i > 4:
            use_time += (time.time() - start_time)

        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1
    if use_mindie:
        mindietorch.finalize()
    infer_num = infer_num - 5
    print(f"average time: {use_time / infer_num:.3f}s\n")

if __name__ == "__main__":
    main()
