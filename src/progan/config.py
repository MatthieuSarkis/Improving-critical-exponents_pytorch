import torch

START_TRAIN_AT_IMG_SIZE = 4
LOGS_PATH = "saved_models/progan/"
LOAD_CHECKPOINT_GEN_PATH = "generator.pt"
LOAD_CHECKPOINT_CRITIC_PATH = "critic.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
DATASET_SIZE = 5000
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 16, 16, 8, 8]
FACTORS = [2**(-n) for n in range(6)]
CHANNELS_IMG = 1
Z_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [50] * len(FACTORS)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4
PERCOLATION_CRITICAL_CONTROL_PARAMETER = 0.5928
CNN_MODEL_PATH = "./saved_models/cnn/2022.02.11.18.36.08/model/final_model.pt"
#CNN_MODEL_PATH = None
CNN_LOSS_RATIO = 0.1