from datetime import datetime
import json
import math
import os
import torch
from argparse import ArgumentParser

from src.progan.progan import ProGan

def main(config: dict) -> None:

    if 'cuda' in config['DEVICE']:
        torch.backends.cudnn.benchmarks = True

    if config['STAT_PHYS_MODEL'] == 'ising':
        config['CONTROL_PARAMETER'] = 2 / math.log(1 + math.sqrt(2))

    save_dir = os.path.join(config["LOGS_PATH"], config["STAT_PHYS_MODEL"], datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
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
        factors=config['FACTORS'],
        path_to_trained_model=None,
        cnn_path=config['CNN_MODEL_PATH'],
        stat_phys_model=config['STAT_PHYS_MODEL'],
        statistical_control_parameter=config['CONTROL_PARAMETER'],
        use_tensorboard=config['USE_TENSORBOARD']
    )

    progan._train(
        start_train_at_img_size=config['START_TRAIN_AT_IMG_SIZE'],
        progressive_epochs=config['PROGRESSIVE_EPOCHS'],
        lambda_gp=config['LAMBDA_GP'],
        batch_sizes=config['BATCH_SIZES'],
        fixed_noise=torch.randn(8, config['NOISE_DIM'], 1, 1).to(config['DEVICE']),
        dataset_size=config['DATASET_SIZE'],
        save_model=config['SAVE_MODEL'],
        cnn_loss_ratio=config['CNN_LOSS_RATIO'] if config['CNN_MODEL_PATH'] is not None else None
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--max_image_size", type=int, default=128)
    args = parser.parse_args()

    n = int(math.log2(args.max_image_size) - 1)

    config = {
        "START_TRAIN_AT_IMG_SIZE": 4,
        "LOGS_PATH": "saved_models/progan/",
        "STAT_PHYS_MODEL": "ising",
        "SAVE_MODEL": True,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "DATASET_SIZE": 20000,
        "LEARNING_RATE": 1e-3,
        "CHANNELS_IMG": 1,
        "NOISE_DIM": 256,
        "IN_CHANNELS": 256,
        "LAMBDA_GP": 10,
        "FACTORS": [2**(-k) for k in range(n)],
        "BATCH_SIZES": [64 for _ in range(n)],
        "PROGRESSIVE_EPOCHS": [80] * (n-1) + [350],
        "CONTROL_PARAMETER": 0.5928,
        #"CNN_MODEL_PATH": "./trained_models_DoNotErase/cnn/2022.02.11.18.36.08/model/final_model.pt",
        "CNN_MODEL_PATH": None,
        "CNN_LOSS_RATIO": 0.1,
        "USE_TENSORBOARD": True
    }

    main(config=config)