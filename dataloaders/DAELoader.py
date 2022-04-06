import torch
from torch.utils.data import Dataset

class DAELoader(Dataset):
    def __init__(self, X, corruption_prob):
        super(DAELoader, self).__init__()

        self.X = torch.Tensor(X)
        self.corruption_prob = corruption_prob
        
    def corrupt(self, x):
        shuffle = torch.randint(0, x.shape[0], size=[x.shape[0]], dtype=int)
        mask = (torch.rand_like(x) < self.corruption_prob).to(torch.float32)

        return (1-mask) * x + mask * x[shuffle]

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, index):
        return self.corrupt(self.X[index, :]), self.X[index]