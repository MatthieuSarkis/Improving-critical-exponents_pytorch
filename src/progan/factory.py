from datetime import datetime
import json
import os
import torch

from src.progan.config import config
from src.progan.progan import ProGan

if 'cuda' in config['DEVICE']:
    torch.backends.cudnn.benchmarks = True

def main():
    
    save_dir = os.path.join(config["LOGS_PATH"], datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    progan = ProGan(
        noise_dim=config['NOISE_DIM'],
        in_channels=config['IN_CHANNELS'],
        channels_img=config['CHANNELS_IMG'],
        learning_rate=config['LEARNING_RATE'],
        device=config['DEVICE'],
        logs_path=save_dir,
        load_model=config['LOAD_MODEL'],
        cnn_path=config['CNN_MODEL_PATH'],
        statistical_control_parameter=config['PERCOLATION_CONTROL_PARAMETER'],
        use_tensorboard=config['USE_TENSORBOARD']
    )

