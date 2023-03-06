from denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
import os

DATASET_SIZE = 100
IMAGE_SIZE = 64
BATCH_SIZE = 16
PERCOLATION_PARAMETER = 0.5928
NUM_TIMESTEPS = 1000
SAMPLING_TIMESTEPS = 250

results_folder = './src/denoising-diffusion-pytorch/ckpt'
os.makedirs(name=results_folder, exist_ok=True)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 3
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = NUM_TIMESTEPS,           # number of steps
    sampling_timesteps = SAMPLING_TIMESTEPS ,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

trainer = Trainer(
    diffusion,
    dataset_size=DATASET_SIZE,
    lattice_size=IMAGE_SIZE,
    percolation_parameter=PERCOLATION_PARAMETER,
    train_batch_size = BATCH_SIZE,
    augment_horizontal_flip = False,
    save_and_sample_every = 1000,
    results_folder = results_folder,
    train_lr = 8e-5,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()