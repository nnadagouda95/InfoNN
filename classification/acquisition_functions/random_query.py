# import libraries

import numpy as np
from .base_func import BaseQuery


class RandomQuery(BaseQuery):
    def __init__(self, model, device, labeled_loader, unlabeled_loader, args):
        super(RandomQuery, self).__init__(model, device, labeled_loader, unlabeled_loader, args)

    def query(self):
        next_samples = np.random.choice(range(len(self.unlabeled_loader.dataset)), size=self.args.n_samples, replace=False)
    
        return next_samples

