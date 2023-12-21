import torch
import gc
from torch.utils.data import DataLoader
from .dataset import LibriDataset
from ..utils import same_seeds, preprocess_data
from ..configs import concat_nframes, valid_ratio, seed, batch_size


def create_dataloader(data_dir, train):
    r"""Create training and validation dataloader or testing dataloader

    Args:
        data_dir (str): `libriphone` folder path
        train (bool): is training or not
    Returns:
        (train_loader, valid_loader) or test_loader
    """
    same_seeds(seed)
    if train:
        # preprocess data
        train_X, train_y, valid_X, valid_y = preprocess_data(data_dir, concat_nframes, train, valid_ratio)

        # get dataset
        train_set = LibriDataset(train_X, train_y)
        valid_set = LibriDataset(valid_X, valid_y)

        # remove raw feature to save memory
        del train_X, train_y, valid_X, valid_y
        gc.collect()

        # get dataloader
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        return train_loader, valid_loader
    else:
        test_X = preprocess_data(data_dir, concat_nframes, train, valid_ratio)
        test_set = LibriDataset(test_X)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return test_loader
