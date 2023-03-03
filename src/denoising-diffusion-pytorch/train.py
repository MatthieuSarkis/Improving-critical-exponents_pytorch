import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from src.data_factory.percolation import generate_percolation_data
import os

IMAGE_SIZE = 32
DATASET_SIZE = 10
BATCH_SIZE = 2
PERCOLATION_PARAMETER = 0.5

X, _ = generate_percolation_data(
    dataset_size=DATASET_SIZE,
    lattice_size=IMAGE_SIZE,
    p_list=[PERCOLATION_PARAMETER],
    split=False,
    save_dir=None
)

X = (X + 1) / 2
#X = X.repeat((1, 1, 1, 1)) # I add fake color channels because I don't find how to use their code with only 1 color channel...

data_folder = './src/denoising-diffusion-pytorch/data'
results_folder = './src/denoising-diffusion-pytorch/results'
os.makedirs(name=data_folder, exist_ok=True)
os.makedirs(name=results_folder, exist_ok=True)

for i in range(X.shape[0]):
    torch.save(obj=X[i], f=os.path.join(data_folder, '{}.pt'.format(i)))

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    #timesteps = 1000,           # number of steps
    timesteps = 1,           # number of steps
    #sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    sampling_timesteps = 1,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

trainer = Trainer(
    diffusion,
    folder=data_folder,
    train_batch_size = BATCH_SIZE,
    augment_horizontal_flip = False,
    #save_and_sample_every = 1000,
    save_and_sample_every = 1,
    results_folder = results_folder,
    train_lr = 8e-5,
    #train_num_steps = 700000,         # total training steps
    train_num_steps = 2,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()