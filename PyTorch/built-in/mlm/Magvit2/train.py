import torch
import argparse

from magvit2_pytorch import (
    VideoTokenizer,
    VideoTokenizerTrainer
)

from torch_npu.contrib import transfer_to_npu

torch.npu.config.allow_internal_format = False

import torch.nn.functional as F
from npu_patch import adaptive_avg_pool2d

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_folder", type=str, default=None, help="Path of training dataset."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for the training dataloader."
    )
    parser.add_argument(
        "--grad_accum_every",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    F.adaptive_avg_pool2d = adaptive_avg_pool2d

    tokenizer = VideoTokenizer(
        image_size = 128,
        init_dim = 64,
        max_dim = 512,
        codebook_size = 1024,
        layers = (
            'residual',
            'compress_space',
            ('consecutive_residual', 2),
            'compress_space',
            ('consecutive_residual', 2),
            'linear_attend_space',
            'compress_space',
            ('consecutive_residual', 2),
            'attend_space',
            'compress_time',
            ('consecutive_residual', 2),
            'compress_time',
            ('consecutive_residual', 2),
            'attend_time',
        ),
        use_gan = False
    )

    trainer = VideoTokenizerTrainer(
        tokenizer,
        dataset_folder = args.dataset_folder,     # folder of either videos or images, depending on setting below
        dataset_type = 'videos',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
        batch_size = args.batch_size,
        grad_accum_every = args.grad_accum_every,
        learning_rate = args.learning_rate,
        num_train_steps = args.num_train_steps
    )

    trainer.train()

if __name__ == "__main__":
    main()
