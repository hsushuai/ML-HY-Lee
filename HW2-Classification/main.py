import warnings
from src.configs import model_path, data_path, preds_path
from src.data_loader import create_dataloader
from src.model import PhonemeClassifier, train_model
from src.utils import predict_and_save
import torch
from colorama import Fore, Style


warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhonemeClassifier().to(device)
train_model(data_path, model)

model = PhonemeClassifier().to(device)
model.load_state_dict(torch.load(model_path))

test_loader = create_dataloader(data_path, False)
predict_and_save(test_loader, model, preds_path)

print(f"\n{Fore.GREEN}🥳 DONE!{Style.RESET_ALL}")
