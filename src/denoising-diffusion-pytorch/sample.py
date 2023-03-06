import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import numpy as np
import os

trained_model_path = 'src/denoising-diffusion-pytorch/logs/2023.03.06.15.06.34/ckpt/model-200.pt'
generated_images_path = 'src/denoising-diffusion-pytorch/logs/2023.03.06.15.06.34/generated_images'
os.makedirs(generated_images_path, exist_ok=True)

IMAGE_SIZE = 32
PERCOLATION_PARAMETER = 0.5928
NUM_TIMESTEPS = 1000
SAMPLING_TIMESTEPS = 250
NUM_SAMPLES = 1000

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

sampled_images = diffusion.sample(batch_size = NUM_SAMPLES).cpu().type(torch.int8).numpy()[:,0,:,:]
np.save(file=os.path.join(generated_images_path, 'L={}_p={}'.format(IMAGE_SIZE, PERCOLATION_PARAMETER)), arr=sampled_images)