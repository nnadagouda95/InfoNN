import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNetSynthetic(nn.Module):
    def __init__(self):
        super(EmbeddingNetSynthetic, self).__init__()

        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 48)
        self.fc3 = nn.Linear(48, 10)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class EmbeddingNetFood(nn.Module):
    def __init__(self):
        super(EmbeddingNetFood, self).__init__()

        self.fcn1 = nn.Linear(6, 12)
        self.fcn2 = nn.Linear(12, 8)
        self.fcn3 = nn.Linear(8, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fcn1(x))
        x = self.relu(self.fcn2(x))
        x = self.fcn3(x)
        return x

class EmbeddingNetAdm(nn.Module):
    def __init__(self):
        super(EmbeddingNetAdm, self).__init__()

        self.fcn1 = nn.Linear(25, 16)
        self.fcn2 = nn.Linear(16, 12)
        self.fcn3 = nn.Linear(12, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fcn1(x))
        x = self.relu(self.fcn2(x))
        x = self.fcn3(x)
        return x
