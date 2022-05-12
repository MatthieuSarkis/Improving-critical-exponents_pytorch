import torch

config = {
    "START_TRAIN_AT_IMG_SIZE": 4,
    "LOGS_PATH": "saved_models/progan/",
    "SAVE_MODEL": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DATASET_SIZE": 5000,
    "LEARNING_RATE": 1e-3,
    "CHANNELS_IMG": 1,
    "NOISE_DIM": 256,
    "IN_CHANNELS": 256,
    "LAMBDA_GP": 10,
    "FACTORS": [2**(-n) for n in range(6)],
    "BATCH_SIZES": [32, 32, 16, 16, 8, 8],
    #"BATCH_SIZES": [32] * 6,
    "PROGRESSIVE_EPOCHS": [50] * 6,
    "NUM_WORKERS": 4,
    "PERCOLATION_CONTROL_PARAMETER": 0.5928,
    "CNN_MODEL_PATH": "./trained_models_DoNotErase/cnn/2022.02.11.18.36.08/model/final_model.pt",
    "CNN_LOSS_RATIO": 0.1,
    "USE_TENSORBOARD": True
}