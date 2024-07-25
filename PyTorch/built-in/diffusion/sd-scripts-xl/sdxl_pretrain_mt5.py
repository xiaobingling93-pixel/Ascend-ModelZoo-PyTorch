# Copyright 2023 Huawei Technologies Co., Ltd
# training with captions

import argparse
import gc
import math
import os
import time
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm
import torch
import torch_npu

try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now.. Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass

try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        from library.ipex import ipex_init

        ipex_init()
except Exception:
    pass
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import sdxl_model_util
from library.sdxl_pretrain_models import SdxlPretrainModelsMT5

import library.train_util as train_util
import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel

torch.npu.config.allow_internal_format = False
UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)

    assert not args.weighted_captions, "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    assert (
            not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
                len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer1= sdxl_train_util.load_tokenizers_mt5(args)

    print("tokenizer1 max_token_length :",tokenizer1.max_token_length)
    
    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        if args.dataset_config is not None:
            print(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                print(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                print("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                print("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1])

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        print(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    print("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model_mt5(args, accelerator, "sdxl", weight_dtype)


    if args.enable_npu_flash_attention:
        accelerator.print("Enable npu flash attention")
        unet.set_use_npu_flash_attention(True)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk,
                                              accelerator.is_main_process)
        vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = args.learning_rate > 0
    train_text_encoder1 = False

    if args.train_text_encoder:
        raise Exception("MT5 text_encoder does not support training")

    text_encoder1.to(weight_dtype)
    text_encoder1.requires_grad_(False)
    text_encoder1.eval()

    # TextEncoderの出力をキャッシュする
    if args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad
        with torch.no_grad(), accelerator.autocast():
            train_dataset_group.cache_text_encoder_outputs(
                (tokenizer1),
                (text_encoder1),
                accelerator.device,
                None,
                args.cache_text_encoder_outputs_to_disk,
                accelerator.is_main_process,
            )
        accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
                args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
    elif args.full_bf16:
        assert (
                args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append(
            {"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})


    # calculate number of trainable parameters
    n_params = 0
    for params in params_to_optimize:
        for p in params["params"]:
            n_params += p.numel()

    accelerator.print(
        f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    sdxlPretrainModels = SdxlPretrainModelsMT5(args, unet, text_encoder1, tokenizer1, weight_dtype)
    sdxlPretrainModels, optimizer = accelerator.prepare(sdxlPretrainModels, optimizer)
    sdxlPretrainModels.to(accelerator.device)

    train_dataloader, lr_scheduler = accelerator.prepare(train_dataloader, lr_scheduler)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # 学習する
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
    accelerator.print(
        f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process,
                        desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name,
                                  init_kwargs=init_kwargs)

    loss_recorder = train_util.LossRecorder()
    

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        sdxlPretrainModels.train()
        profiler = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                          profile_type=os.getenv('PROFILE_TYPE'))
        step_end_time = time.time()
        for step, batch in enumerate(train_dataloader):
            profiler.start()
            step_data_time = time.time() - step_end_time
            current_step.value = global_step
            with accelerator.accumulate(sdxlPretrainModels):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # latentに変換
                        latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)
                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                latents = latents * sdxl_model_util.VAE_SCALE_FACTOR
                noise_pred, noise, timesteps = sdxlPretrainModels(args, batch, accelerator, noise_scheduler, latents)

                target = noise

                if (
                        args.min_snr_gamma
                        or args.scale_v_pred_loss_like_noise_pred
                        or args.v_pred_like_loss
                        or args.debiased_estimation_loss
                ):
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    optimizer.clip_grad_norm(args.max_grad_norm)


                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                profiler.end()

            #Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    [tokenizer1],
                    [text_encoder1],
                    unet,
                )



            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler,
                                            args.optimizer_type)  # U-Net is included in block_lrs

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"current_loss": current_loss, "avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            step_train_time = time.time() - step_end_time
            accelerator.print(f"step_train_time: {step_train_time}")
            accelerator.print(f"step_data_time: {step_data_time}")
            accelerator.print(
                f"FPS: {args.train_batch_size * accelerator.num_processes / (step_train_time - step_data_time)}")

            step_end_time = time.time()


        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()


    accelerator.end_training()

    if args.save_state:  # and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す




def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )

    parser.add_argument("--diffusers_xformers", action="store_true",
                        help="use xformers by diffusers / Diffusersでxformersを使用する")
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
             + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument(
        "--mt5_encoder_path",
        type=str,
        default=None,
        help=f"mt5 encoder path for SDXL",
    )
    parser.add_argument(
        "--mt5_tokenizer_path",
        type=str,
        default=None,
        help=f"mt5 tokenizer path for SDXL",
    )
    parser.add_argument(
        "--mt5_dim_size",
        type=int,
        default=4096,
        help=f"mt5 dim size",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
