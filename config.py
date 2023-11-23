# Description: Configuration file for the project
# Author: Alikhan Nurkamal, Nail Fakhrutdinov
# Last Modified: 2023-11-23

# Config variables
config = {
    'ROOT_DIR': 'data/',
    'TEST_DIR': 'data/test/',
    'MODELS_DIR': 'models/',
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 100,
    'PATIENCE': 7,
    'LR_INIT': 1e-4,  # pls change this
    'LR_FINAL': 1e-4,  # pls change this
    'WEIGHT_DECAY': 5e-4,
    'NUM_CLASSES': 10,
    'NUM_WORKERS': 4,
    'NUM_CHANNELS': 3,
    'DEVICE': 'cuda',
}
