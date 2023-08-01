# import libraries

import os, sys
import torch.nn as nn
import torch.nn.functional as F
from .consistent_mc_dropout import *


class MNIST_model(BayesianModule):
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        feature = self.fc1(input)
        input = F.relu(self.fc1_drop(feature))
        logits = self.fc2(feature)
        output = F.log_softmax(logits, dim=1)

        return feature, output
        
        



    
