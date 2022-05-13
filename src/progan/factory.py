from datetime import datetime
import json
import os
import torch

from src.progan.progan import ProGan

def main(
    path_to_trained_model: str,
    n_images: int = 10
) -> None:

    with open(os.path.join(path_to_trained_model, 'config.json')) as f:
        config = json.load(f)
    
    if 'cuda' in config['DEVICE']:
        torch.backends.cudnn.benchmarks = True
    
    save_dir = os.path.join('generated_data', 'model_progan_' + path_to_trained_model.split('/')[-1])
    os.makedirs(save_dir, exist_ok=True)

    progan = ProGan(
        noise_dim=config['NOISE_DIM'],
        in_channels=config['IN_CHANNELS'],
        channels_img=config['CHANNELS_IMG'],
        learning_rate=config['LEARNING_RATE'],
        device='cpu',
        logs_path=save_dir,
        path_to_trained_model=path_to_trained_model,
        cnn_path=None,
        statistical_control_parameter=config['PERCOLATION_CONTROL_PARAMETER'],
        use_tensorboard=False
    )

    progan.generate_images(
        steps=len(config['FACTORS'])-1,
        n_images=n_images
    )

if __name__ == '__main__':

    main(
        path_to_trained_model='./trained_models_DoNotErase/progan/2022.05.13.13.43.18',
        n_images=100000
    )
