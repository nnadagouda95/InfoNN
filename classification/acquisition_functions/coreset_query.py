# import libraries

import torch
import numpy as np
from .base_func import BaseQuery


class CoreSetQuery(BaseQuery):
    def __init__(self, model, device, labeled_loader, unlabeled_loader, args):
        super(CoreSetQuery, self).__init__(model, device, labeled_loader, unlabeled_loader, args)

    def query(self):
        # compute the embeddings
        embeddings_u, embeddings_l, _ = self.get_embedding()

        next_samples = []
        for _ in range(self.args.n_samples):
            distances = torch.cdist(embeddings_l, embeddings_u)
            min_dist = torch.min(distances, dim = 0, keepdim = True)[0]
            ind = torch.argmax(min_dist)
            next_samples.append(ind)
            embeddings_l = torch.cat((embeddings_l,embeddings_u[ind].reshape(1,-1)))
            embeddings_u = torch.cat((embeddings_u[:ind],embeddings_u[ind+1:]))

        next_samples = torch.stack(next_samples)
        next_samples = next_samples.cpu().detach().numpy() 
        next_samples = np.ndarray.flatten(np.squeeze(next_samples))

        return next_samples

