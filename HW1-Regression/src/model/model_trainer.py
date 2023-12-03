import math
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ..configs import CONFIG, DEVICE, BASE_DIR
from tqdm import tqdm


def trainer(train_loader, valid_loader, model):
    """Train the model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)

    n_epochs, best_loss, step, early_stop_count = CONFIG["n_epochs"], math.inf, 0, 0

    save_model_dir = os.path.join(BASE_DIR, "models")
    save_model_path = os.path.join(save_model_dir, "model.ckpt")
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    writer = SummaryWriter()

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            step += 1

            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar("Loss/train", mean_train_loss, step)

        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f"Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}")
        writer.add_scalar("Loss/valid", mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            early_stop_count = 0
            torch.save(model.state_dict(), save_model_path)
            print(f"Saving model with loss {mean_valid_loss:.3f}...")
        else:
            early_stop_count += 1

        if early_stop_count >= CONFIG["early_stop"]:
            print("\nModel is not improving, so we halt the training session.")
            return
