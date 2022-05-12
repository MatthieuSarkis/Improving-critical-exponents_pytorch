from datetime import datetime
import json
import os
import torch

from src.progan.progan import ProGan

def main(
    path_to_trained_model: str,
    n_images: int = 10
) -> None:

    #with open(os.path.join(path_to_trained_model, 'config.json')) as f:
    #    config = json.load(f)
    #
    #if 'cuda' in config['DEVICE']:
    #    torch.backends.cudnn.benchmarks = True
    #
    #save_dir = os.path.join('generated_data', datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    #os.makedirs(save_dir, exist_ok=True)
#
    #progan = ProGan(
    #    noise_dim=config['NOISE_DIM'],
    #    in_channels=config['IN_CHANNELS'],
    #    channels_img=config['CHANNELS_IMG'],
    #    learning_rate=config['LEARNING_RATE'],
    #    device=config['DEVICE'],
    #    logs_path=save_dir,
    #    load_model=config['LOAD_MODEL'],
    #    cnn_path=None,
    #    statistical_control_parameter=config['PERCOLATION_CONTROL_PARAMETER'],
    #    use_tensorboard=False
    #)

    #progan.generate_images(
    #    steps=len(config['FACTORS']),
    #    n_images=n_images
    #)

    save_dir = os.path.join('generated_data', datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    progan = ProGan(
        noise_dim=256,
        in_channels=256,
        channels_img=1,
        learning_rate=1e-3,
        device='cpu',
        logs_path=save_dir,
        path_to_trained_model=path_to_trained_model,
        cnn_path=None,
        statistical_control_parameter=0.5928,
        use_tensorboard=False
    )

    progan.generate_images(
        steps=5,
        n_images=n_images
    )

if __name__ == '__main__':

    main(
        path_to_trained_model='./trained_models_DoNotErase/progan/2022.05.11.18.01.04',
        n_images=10
    )

