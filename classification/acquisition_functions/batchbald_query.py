# import libraries

from __future__ import print_function
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from itertools import combinations_with_replacement
from .base_func import BaseQuery


"""
Code from: https://github.com/IzzyBrand/PyTorch_BatchBALD/blob/master/util.py
"""

class BatchBALDQuery(BaseQuery):
    def __init__(self, model, device, labeled_loader, unlabeled_loader, args):
        super(BatchBALDQuery, self).__init__(model, device, labeled_loader, unlabeled_loader, args)

    def query(self):
        
        m = 1e3  # number of MC samples for label combinations
        k = 100
        processing_batch_size = 128

        num_sub_pool = self.args.num_sub_pool_batchbald  # number of datapoints in the subpool from which we acquire
        c = self.args.n_classes # number of classes

        # performing BatchBALD on the whole pool is very expensive, so we take
        # a random subset of the pool.
        pool_data = self.unlabeled_loader.dataset
        num_extra = len(pool_data) - num_sub_pool
        if num_extra > 0:
            sub_pool_data, _ = random_split(pool_data, [num_sub_pool, num_extra])
        else:
            # even if we don't have enough data left to split, we still need to
            # call random_split to avoid messing up the indexing later on
            sub_pool_data, _ = random_split(pool_data, [len(pool_data), 0])

         # forward pass on the pool once to get class probabilities for each x
        with torch.no_grad():
            pool_loader = torch.utils.data.DataLoader(sub_pool_data,
                batch_size=processing_batch_size, pin_memory=True, shuffle=False)
            pool_p_y = torch.zeros(len(sub_pool_data), c, k)

            self.model.eval()
            for batch_idx, (data, _) in enumerate(pool_loader):
                end_idx = batch_idx + data.shape[0]
    #             print(batch_idx)
    #             print(data.shape[0])
    #             print(end_idx)
                pool_p_y[batch_idx:end_idx] = torch.exp(self.model(data.to(self.device, dtype=torch.float),k)[1]).permute(0,2,1)

        # this only need to be calculated once so we pull it out of the loop
        H2 = (H(pool_p_y).sum(axis=(1,2))/k).to(self.device)

        # get all class combinations
        c_1_to_n = class_combinations(c, self.args.n_samples, m)

        # tensor of size [m x k]
        p_y_1_to_n_minus_1 = None

        # store the indices of the chosen datapoints in the subpool
        best_sub_local_indices = []
        # create a mask to keep track of which indices we've chosen
        remaining_indices = torch.ones(len(sub_pool_data), dtype=bool).to(self.device)
        for n in range(self.args.n_samples):
            # tensor of size [N x m x l]
            p_y_n = pool_p_y[:, c_1_to_n[:, n], :].to(self.device)
            # tensor of size [N x m x k]
            p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n)\
                if p_y_1_to_n_minus_1 is not None else p_y_n

            # and compute the left entropy term
            H1 = H(p_y_1_to_n.mean(axis=2)).sum(axis=1)
            # scores is a vector of scores for each element in the pool.
            # mask by the remaining indices and find the highest scoring element
            scores = H1 - H2
            # print(scores)
            best_local_index = torch.argmax(scores - np.inf*(~remaining_indices)).item()
            # print(f'Best idx {best_local_index}')
            best_sub_local_indices.append(best_local_index)
            # save the computation for the next batch
            p_y_1_to_n_minus_1 = p_y_1_to_n[best_local_index]
            # remove the chosen element from the remaining indices mask
            remaining_indices[best_local_index] = False

    #     print('Remaining indices: {}'.format(remaining_indices))
    #     print('Selected indices: {}'.format(best_sub_local_indices))

        # we've subset-ed our dataset twice, so we need to go back through
        # subset indices twice to recover the global indices of the chosen data
        best_local_indices = np.arange(len(sub_pool_data))[best_sub_local_indices]
        next_samples = np.arange(len(pool_data))[best_local_indices]

        return next_samples


def class_combinations(c, n, m=np.inf):
    """ Generates an array of n-element combinations where each element is one of
    the c classes (an integer). If m is provided and m < n^c, then instead of all
    n^c combinations, m combinations are randomly sampled.

    Arguments:
        c {int} -- the number of classes
        n {int} -- the number of elements in each combination

    Keyword Arguments:
        m {int} -- the number of desired combinations (default: {np.inf})

    Returns:
        np.ndarry -- An [m x n] or [n^c x n] array of integers in [0, c)
    """

    if m < c**n:
        # randomly sample combinations
        return np.random.randint(c, size=(int(m), n))
    else:
        p_c = combinations_with_replacement(np.arange(c), n)
        return np.array(list(iter(p_c)), dtype=int)

def H(x, eps=1e-6):
    """ Compute the element-wise entropy of x

    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)

    Keyword Arguments:
        eps {float} -- prevent failure on x == 0

    Returns:
        torch.Tensor -- H(x)
    """
    return -(x+eps)*torch.log(x+eps)

