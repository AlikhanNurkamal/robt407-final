# Description: Configuration file for the project
# Author: Alikhan Nurkamal, Nail Fakhrutdinov
# Last Modified: 2023-11-24
import torch

# Config variables
config = {
    'TRAIN_DIR': 'data/train/',
    'TEST_DIR': 'data/test/',
    'MODELS_DIR': 'models/',
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 100,
    'PATIENCE': 9,
    'LR_INIT': 1e-4,
    'WEIGHT_DECAY': 5e-4,
    'NUM_CLASSES': 10,
    'NUM_WORKERS': 4,
    'NUM_CHANNELS': 3,
    'DEVICE': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
}
