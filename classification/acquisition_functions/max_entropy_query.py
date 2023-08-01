# import libraries

import torch
import numpy as np
from .base_func import BaseQuery


class MaxEntropyQuery(BaseQuery):
    def __init__(self, model, device, labeled_loader, unlabeled_loader, args):
        super(MaxEntropyQuery, self).__init__(model, device, labeled_loader, unlabeled_loader, args)

    def query(self):
        # compute the class probablities
        probabilities = []

        self.model.eval()
        with torch.no_grad():
            for data, _ in self.unlabeled_loader:
                _, output = self.model(data.to(self.device, dtype=torch.float),1)
                output = output.squeeze(1)
                prob = torch.exp(output)
                probabilities.append(prob)

        probabilities = torch.cat(probabilities)
        entropies = -torch.sum(probabilities * torch.log2(probabilities), dim=1)
        entropies = entropies.cpu().detach().numpy() 
        len = entropies.shape[0]
        k = len - self.args.n_samples
        indices = np.argpartition(entropies, k)
        next_samples = indices[k:]

        next_samples = np.ndarray.flatten(np.squeeze(next_samples))

        return next_samples

