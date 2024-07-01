num_frames = 16
fps = 24 // 3
image_size = (256, 256)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    enable_flashattn=False,
    enable_layernorm_kernel=False,
    from_pretrained="PRETRAINED_MODEL",
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
)
dtype = "fp32"

# Others
batch_size = 1
seed = 42
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./outputs/samples/"
vae_path = "stabilityai/sd-vae-ft-ema"
t5_path = "./t5-v1_1-xxl"
use_mindie = 1
device_id = 0
output_dir="./models"
