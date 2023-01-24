import torch.nn.functional as F
from torch import nn


class ColumnFC(nn.Module):
    def __init__(self, input_dim=100, output_dim=100, dropout=0.1):
        super(ColumnFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x