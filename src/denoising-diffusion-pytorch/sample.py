import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import numpy as np
import os
from math import log, sqrt

log_path = 'src/denoising-diffusion-pytorch/logs/ising/2023.03.06.23.10.08'

trained_model_path = os.path.join(log_path, '/ckpt/model-50.pt')
generated_images_path = os.path.join(log_path, 'generated_images')
os.makedirs(generated_images_path, exist_ok=True)

args = {
    'NUM_SAMPLES': 5000,
    'IMAGE_SIZE': 32,
    'STATISTICAL_PARAMETER': 2/log(1 + sqrt(2)),
    'NUM_TIMESTEPS': 1000,
    'SAMPLING_TIMESTEPS': 250,
}

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 3
)

diffusion = GaussianDiffusion(
    model,
    image_size = args['IMAGE_SIZE'],
    timesteps = args['NUM_TIMESTEPS'],           # number of steps
    sampling_timesteps = args['SAMPLING_TIMESTEPS'] ,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

sampled_images = diffusion.sample(batch_size = args['NUM_SAMPLES']).cpu().type(torch.int8).numpy()[:,0,:,:]
np.save(file=os.path.join(generated_images_path, 'L={}_p={:.4f}'.format(args['IMAGE_SIZE'], args['STATISTICAL_PARAMETER'])), arr=sampled_images)