import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpTripletLoss(nn.Module):
    def __init__(self, scaling=1):
        super(ExpTripletLoss, self).__init__()
        self.scaling = scaling
        self.eps = 1e-9

    def forward(self, dist_pos, dist_neg, reduction="sum"):
        losses = torch.exp(-self.scaling*(dist_neg - dist_pos))

        return losses.sum() if (reduction == "sum") else losses.mean()

class MSESoftmaxLoss(nn.Module):
    def __init__(self):
        super(MSESoftmaxLoss, self).__init__()
        self.eps = 1e-9

    def forward(self, dist_pos, dist_neg, reduction="sum"):
        losses = torch.norm(F.softmax(torch.stack((dist_pos, dist_neg)).T) - torch.tensor([0,1]), dim=1).pow(2)
        return losses.sum() if (reduction == "sum") else losses.mean()

class LogProbLoss(nn.Module):
    def __init__(self, mu):
        super(LogProbLoss, self).__init__()
        self.mu = mu
        self.eps=1e-9

    def forward(self, dist_pos, dist_neg, reduction="sum"):
        num = dist_neg + self.mu
        den = dist_neg + dist_pos + 2*self.mu
        losses = -torch.log(num/(den + self.eps))

        return losses.sum() if (reduction == "sum") else losses.mean()

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, dist_pos, dist_neg, reduction = "sum"):
        delta = (dist_neg - dist_pos)
        losses = torch.maximum(1-delta, torch.tensor([0]))
        return losses.sum() if (reduction == "sum") else losses.mean()
