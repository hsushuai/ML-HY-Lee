import torch
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: Improve the `config`
CONFIG = {
    'seed': 8590,  # Your seed number, you can pick your lucky number. :)
    'select_all': False,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 5000,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 600,  # If model has not improved for this many consecutive epochs, stop training.
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
