import torch
from torch.utils.data import Dataset


class MatrixLoader(Dataset):
    """Simple dataloader given the data matrix X and response y
    batch is assumed to be the first dimension"""
    def __init__(self, X, y, classification=True):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

        if classification:
            self.y = self.y.to(int)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]