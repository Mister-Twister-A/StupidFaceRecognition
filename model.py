import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TwinModel(nn.Module):
    def __init__(self):
        super(TwinModel, self).__init__()
        self.fc1 = nn.Linear()

    def forward(self, x):
        pass