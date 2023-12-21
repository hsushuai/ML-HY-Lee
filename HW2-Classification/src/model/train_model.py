from ..data_loader import create_dataloader
from .model import PhonemeClassifier
from torch import nn
import torch
from ..configs import *
import os
from tqdm import tqdm
from colorama import Fore, Style
from torch.utils.tensorboard import SummaryWriter


def train_model(dataset_path, model):
    r"""Train the model.

    Args:
        dataset_path (str): Path to the `libriphone` dataset.
        model (torch.nn.Module): Model to train.
    """
    train_loader, val_loader = create_dataloader(dataset_path, True)

    if not os.path.isdir(os.path.dirname(model_path)):
        os.mkdir(os.path.dirname(model_path))

    writer = SummaryWriter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(num_epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train()
        for x, y in tqdm(train_loader, desc="Training    "):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs, 1)
            train_acc += (pred.detach() == y.detach()).sum().item()
            train_loss += loss.item()

        # validating
        model.eval()
        for x, y in tqdm(val_loader, desc="Validating  "):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                outputs = model(x)
                loss = criterion(outputs, y)

            _, pred = torch.max(outputs, 1)
            val_acc += (pred.cpu() == y.cpu()).sum().item()
            val_loss += loss.item()

        writer.add_scalar("train_loss", train_loss/len(train_loader), epoch)
        writer.add_scalar("val_loss", val_loss/len(val_loader), epoch)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]: Train Acc: {train_acc/len(train_loader.dataset):.5f} Loss: {train_loss/len(train_loader):.5f} | Val Acc: {val_acc/len(val_loader.dataset):.5f} loss: {val_loss/len(val_loader):.5f}")

        # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Saving model with acc {best_acc / len(val_loader.dataset):.5f}...')

    print(f'\n{Fore.GREEN}Training completed!{Style.RESET_ALL} Best validating accuracy is {best_acc / len(val_loader.dataset):.5f}\n')


























