from torch import nn


class PhonemeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Modify the model structure
        self.net = nn.Sequential(nn.LazyLinear(128),
                                 nn.ReLU(),
                                 nn.LazyLinear(256),
                                 nn.ReLU(),
                                 nn.LazyLinear(256),
                                 nn.ReLU(),
                                 nn.LazyLinear(41))

    def forward(self, x):
        return self.net(x)
