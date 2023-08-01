# import libraries

from __future__ import print_function
import argparse
import torch
import numpy as np
from .utils import kmeans
import math
from .base_func import BaseQuery


def form_queries(embeddings_u, embeddings_l, labels, n_classes, num_neighbors):
        
    """
    Find the nearest labeled samples from every class to each unlabeled sample
    
    Arguments:
        embeddings_u: embeddings corresponding to the unlabeled samples, of shape (n_u, d)
        embeddings_l: embeddings corresponding to the unlabeled samples, of shape (n_l, d)
        labels: labels corresponding to the labeled samples
        n_classes: number of classes in the dataset
        num_neighbors: number of neighbors to be considered while constructing the queries
    
    Returns:
        queries: matrix of shape (n_u, num_neighbors) containing distances between unlabeled samples 
                           and the nearest neighbors 
        dist_std: standard deviation of all the distances 
        mu: hyperparameter for the probability model
    """
    
    distances = torch.cdist(embeddings_l, embeddings_u)
    dist_std = torch.std(distances)
    print('Dist std: {}'.format(dist_std))

    mu = torch.max(distances)

    print('mu: {}'.format(mu))
    distances = torch.cat((labels, distances),1)
    _, ind = torch.topk(distances[:,1:], topk, dim = 0, largest=False)
    neighbours = distances[:,0][ind]

    candidate_queries = []

    for k in range(n_classes):
        mask = distances[:,0] == k
        nearest_neighbors = torch.min(distances[torch.nonzero(mask, as_tuple=True)][:,1:], dim = 0, keepdim = True)
        candidate_queries.append(nearest_neighbors[0])

    candidate_queries = torch.cat(candidate_queries).t()
    
    queries, _ = torch.topk(candidate_queries, num_neighbors, dim = 1, largest=False)
    
    return queries, dist_std, mu


def class_probabilities(distances, mu):
    """
    Compute the class probabilities
    
    Arguments:
        distances: distances between unlabeled samples and nearest neighbors
        mu: regularization parameter for the probability model
    Returns:
        prob: the probability corresponding to the respective classes
    """
    
    prob = 1 / (distances**2 + mu)
    prob = prob / torch.sum(prob, dim=1, keepdim=True)
        
    return prob


def mutual_information(device, query, num_samples, dist_std, mu):
    """
    Compute mutual information bwtween a candidate query and the embedding
    
    Arguments:
        device: device used for computations
        query: a candidate query
        num_samples: Number of samples generated from the distribution
        dist_std: variance parameter for the distribution
        mu: Optional regularization parameter for the probabilistic r60086esponse model
    Returns:
        information: mutual information 
    """
    
    distances = []
        
    for i in range(query.shape[0]):
        
        distances.append(torch.abs(torch.normal(query[i].float(), dist_std, (num_samples,1))))
        
    distances = torch.squeeze(torch.stack(distances, dim=2), dim=1)
    distances = distances.to(device)
    
    probability_samples = class_probabilities(distances, mu)
    entropy_samples = -torch.sum(probability_samples * torch.log2(probability_samples), dim=1)
    expected_probabilities = torch.sum(probability_samples, dim=0, keepdim=True) / num_samples
    entropy  = -torch.sum(expected_probabilities * torch.log2(expected_probabilities))
    expected_entropy = torch.sum(entropy_samples) / num_samples
    information = entropy - expected_entropy
    
    return information

class InfoNNQuery(BaseQuery):
    def __init__(self, model, device, labeled_loader, unlabeled_loader, args):
        super(InfoNNQuery, self).__init__(model, device, labeled_loader, unlabeled_loader, args)

    def query(self):
        
        # distribution parameters
        num_samples = 100
        n_samples = self.args.n_samples
        print('distribution samples: {}'.format(num_samples))

        # compute the embeddings
        embeddings_u, embeddings_l, labels = self.get_embedding()

        # generate the candidate queries
        candidate_queries, dist_std, mu = form_queries(embeddings_u, embeddings_l, labels, self.args.n_classes, self.args.num_neighbors_info_nn, self.args.topk_info_nn)

        # select the optimal query
        infogain_u = []

        for i in range(candidate_queries.shape[0]):
            temp = mutual_information(self.device, candidate_queries[i], num_samples, dist_std, mu)
            infogain_u.append(temp)

        infogain_u = torch.stack(infogain_u)

        num_clusters = 10

        # apply kmeans clustering
        cluster_ids, cluster_centers = kmeans(X=embeddings_u, num_clusters=num_clusters, distance='euclidean', device=self.device)
        samples_u = torch.cat((cluster_ids.float().reshape(-1,1), infogain_u.reshape(-1,1)), dim=1)

        next_samples = []
        num_unlabeled = cluster_ids.shape[0]
        for k in range(num_clusters):
            mask = samples_u[:,0] == k
            true_ind = torch.nonzero(mask, as_tuple=True)
            cluster_size = true_ind[0].size()[0]
            num_per_cluster = math.ceil((cluster_size * n_samples) / num_unlabeled)
            _, pseudo_ind = torch.topk(samples_u[true_ind][:,1], num_per_cluster, largest=True)
            topk_true_ind = true_ind[0][pseudo_ind]
            next_samples.append(topk_true_ind)
        next_samples = torch.cat(next_samples)
        next_samples = next_samples.cpu().detach().numpy()
        next_samples = np.ndarray.flatten(np.squeeze(next_samples))
        next_samples = next_samples[:n_samples]

        return next_samples


