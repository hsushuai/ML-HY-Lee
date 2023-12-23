from torch import nn


class PhonemeClassifier(nn.Module):
    def __init__(self, num_blocks=4, dropout=0.5):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout)
            ) for _ in range(num_blocks)
        ])
        self.linear = nn.LazyLinear(41)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return self.linear(x)
