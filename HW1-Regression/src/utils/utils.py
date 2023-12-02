import csv
import torch
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    r"""Split provided training data into training and validation set
    :return (train_set, valid_set)"""
    valid_set_size = int(len(data_set) * valid_ratio)
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set,
                                        [train_set_size, valid_set_size],
                                        torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    """Predict on test set"""
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def save_pred(preds, file):
    """ Save predictions to specified file """
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
