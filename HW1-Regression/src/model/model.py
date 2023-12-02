from torch import nn


class COVIDForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Modify model's structure
        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
