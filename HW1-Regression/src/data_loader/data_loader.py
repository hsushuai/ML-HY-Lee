import pandas as pd
from ..utils import train_valid_split, same_seed
from .select_feat import select_feat
from ..configs import CONFIG
from .dataset import COVID19Dataset
from torch.utils.data import DataLoader


def create_dataloader(train_data_path, test_data_path):
    r"""Creat training, validation and test data loader
    :return (train_loader, valid_loader, test_loader)"""

    # Set seed for reproducibility
    same_seed(CONFIG['seed'])

    # train_data size: 3009 x 89 (35 states + 18 features x 3 days)
    # test_data size: 997 x 88 (without last day's positive rate)
    train_data, test_data = pd.read_csv(train_data_path).values, pd.read_csv(test_data_path).values

    train_data, valid_data = train_valid_split(train_data, CONFIG['valid_ratio'], CONFIG['seed'])

    # Print out the data size
    print(f"""train_data size: {train_data.shape}
    valid_data size: {valid_data.shape}
    test_data size: {test_data.shape}""")

    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, CONFIG['select_all'])

    # Print out the number of features
    print(f"number of features: {x_train.shape[1]}")

    train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
        COVID19Dataset(x_valid, y_valid), \
        COVID19Dataset(x_test)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, pin_memory=True)
    return train_loader, valid_loader, test_loader
