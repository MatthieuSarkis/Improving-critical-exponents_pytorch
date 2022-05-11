import torch

import src.progan.config as config
from src.progan.progan import ProGan

if 'cuda' in config.DEVICE:
    torch.backends.cudnn.benchmarks = True

def main():
    
    progan = ProGan(
        z_dim=config.Z_DIM,
        in_channels=config.IN_CHANNELS,
        channels_img=config.CHANNELS_IMG,
        learning_rate=config.LEARNING_RATE,
        device=config.DEVICE,
        logs_path=config.LOGS_PATH,
        load_model=config.LOAD_MODEL,
        cnn_path=config.CNN_MODEL_PATH
    )

    progan._train(
        start_train_at_img_size=config.START_TRAIN_AT_IMG_SIZE,
        progressive_epochs=config.PROGRESSIVE_EPOCHS,
        lambda_gp=config.LAMBDA_GP,
        fixed_noise=config.FIXED_NOISE,
        dataset_size=config.DATASET_SIZE,
        save_model=config.SAVE_MODEL,
        cnn_loss_ratio=config.CNN_LOSS_RATIO if config.CNN_MODEL_PATH is not None else None
    )

if __name__ == "__main__":

    main()