from denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
import os
from datetime import datetime
import json

args = {
    'DATASET_SIZE': 15000,
    'IMAGE_SIZE': 32,
    'BATCH_SIZE': 128,
    'PERCOLATION_PARAMETER': 0.5928,
    'NUM_TIMESTEPS': 1000,
    'SAMPLING_TIMESTEPS': 250,
    'TRAIN_NUM_STEPS': 10000,
    'SAVE_AND_SAMPLE_EVERY': 50,
    'TRAIN_LR': 8e-5,
    'EMA_DECAY': 0.995
}

results_folder = os.path.join('./src/denoising-diffusion-pytorch/logs/', datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
os.makedirs(name=os.path.join(results_folder, 'ckpt'), exist_ok=True)

with open(os.path.join(results_folder, 'args.json'), 'w') as f:
    json.dump(args, f, indent=4)

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

trainer = Trainer(
    diffusion,
    dataset_size=args['DATASET_SIZE'],
    lattice_size=args['IMAGE_SIZE'],
    percolation_parameter=args['PERCOLATION_PARAMETER'],
    train_batch_size = args['BATCH_SIZE'],
    augment_horizontal_flip = False,
    save_and_sample_every = args['SAVE_AND_SAMPLE_EVERY'],
    results_folder = os.path.join(results_folder, 'ckpt'),
    train_lr = args['TRAIN_LR'],
    train_num_steps = args['TRAIN_NUM_STEPS'],        # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = args['EMA_DECAY'],                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()