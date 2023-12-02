import torch
from src import COVIDForecaster, trainer, DEVICE, create_dataloader, predict, save_pred


model = COVIDForecaster().to(DEVICE)
train_loader, valid_loader, test_loader = create_dataloader("./data/covid_train.csv", "./data/covid_test.csv")
trainer(train_loader, valid_loader, model)

model = COVIDForecaster().to(DEVICE)
model.load_state_dict(torch.load("./models/model.ckpt"))
preds = predict(test_loader, model, DEVICE)
save_pred(preds, "./data/pred.csv")

# Open terminal and run `tensorboard --logdir runs` to visualize training process
