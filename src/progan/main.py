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
        path_to_trained_model=None,
        cnn_path=config['CNN_MODEL_PATH'],
        statistical_control_parameter=config['PERCOLATION_CONTROL_PARAMETER'],
        use_tensorboard=config['USE_TENSORBOARD']
    )

    progan._train(
        start_train_at_img_size=config['START_TRAIN_AT_IMG_SIZE'],
        progressive_epochs=config['PROGRESSIVE_EPOCHS'],
        lambda_gp=config['LAMBDA_GP'],
        fixed_noise=torch.randn(8, config['NOISE_DIM'], 1, 1).to(config['DEVICE']),
        dataset_size=config['DATASET_SIZE'],
        save_model=config['SAVE_MODEL'],
        cnn_loss_ratio=config['CNN_LOSS_RATIO'] if config['CNN_MODEL_PATH'] is not None else None
    )

if __name__ == "__main__":

    main()