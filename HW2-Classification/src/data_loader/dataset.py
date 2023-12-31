from torch.utils.data import Dataset


class LibriDataset(Dataset):
    def __init__(self, features, labels=None):
        super().__init__()
        self.features = features
        if labels is None:
            self.labels = None
        else:
            self.labels = labels

    def __getitem__(self, index):
        if self.labels is not None:
            return self.features[index], self.labels[index]
        else:
            return self.features[index]

    def __len__(self):
        return len(self.features)
