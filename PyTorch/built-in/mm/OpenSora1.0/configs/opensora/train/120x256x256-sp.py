num_frames = 120
frame_interval = 3
image_size = (256, 256)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2-seq"
sp_size = 8 # 当sp_size设置为1时，将不会使能序列并行
# context_parallel_algo设置为'megatron_cp_algo'表示序列并行使用ring attention算法, 设置为"ulysses_cp_algo"表示序列并行算法使用ulysses算法
context_parallel_algo = 'megatron_cp_algo'
use_cp_send_recv_overlap = True  # 是否开启序列并行send recv overlap, 仅在context_parallel_algo设置为'megatron_cp_algo'有效
# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="PixArt-XL-2-512x512.pth",
    enable_layernorm_kernel=True,
    enable_flashattn=True,
    enable_sequence_parallelism=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 1
ckpt_every = 100000
load = None

batch_size = 1
lr = 2e-5
grad_clip = 1.0
