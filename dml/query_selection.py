import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.linalg import svdvals
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb
from itertools import permutations, combinations, chain
from functools import reduce

from utils import *

from collections import defaultdict
import random
from scipy import linalg
from scipy.special import comb
from copy import copy
from random import shuffle

import math
import pickle
import datetime
import time
import os

def info_NN_queries_emb(N, B, K, Z, query_pool, n_samples = 100, dist_std = 0.5, mu = 1e-6, B_tilde = 3):

    queries_v, mi_v, queries = {}, [], []

    if B < N:
        min_B = 1000
        min_q = []
        for head in range(N):
            bodies = query_pool[head]
            MI_h = mutual_information_nn_emb(Z, head, bodies, n_samples=n_samples, dist_std=dist_std, mu=mu)
            if head < np.minimum(B_tilde, B):
                max_MI = np.amax(MI_h)
                query = (head,) + bodies[np.argmax(MI_h)]
                queries_v[query] = max_MI
                if max_MI < min_B:
                    min_B = max_MI
                    min_q = query

                continue

            if np.amax(MI_h) > min_B:
                max_MI = np.amax(MI_h)
                queries_v[(head,) + bodies[np.argmax(MI_h)]] = max_MI
                del queries_v[min_q]

                min_q = min(queries_v, key=queries_v.get)
                min_B = queries_v[min_q]

    B_tilde = np.minimum(B_tilde, B)

    for key in queries_v.keys():
        query = list(key)
        queries.append(query)
        head = query[0]
        query_pool[head].remove(tuple(query[1:]))

    return queries


def info_NN_queries_dist(N, B, K, Z, query_pool, dist_std = 5, B_tilde = 3, mu=1e-6, subsample = False):
    heads = list(range(N))
    if subsample:
        ind = np.random.choice(N, 1000, replace=False)
        heads = [heads[i] for i in ind]

    queries_v = []
    mi_v = []
    ct = 0
    if B < N:
        for head in heads:
            if query_pool[head]:
                nn_body, ig_max = primal_body_selector_nn(Z, head, query_pool[head], dist_std = dist_std, mu=mu)
                nn_tuple = [head] + list(nn_body)

                emb_dist = 0

                mi_v.append(ig_max + emb_dist)
                queries_v.append(nn_tuple)
                ct+=1
    else:
        for head in heads:
            if query_pool[head]:
                _, infogains = primal_body_selector_nn(Z, head, query_pool[head], mu=mu, full=True)
                queries_v += [[head] + list(query_pool[head][i]) for i in range(len(query_pool[head]))]
                mi_v += infogains

                ct += 1

    queries = []
    B_tilde = np.minimum(B_tilde, B)
    ind = np.argpartition(mi_v, -B_tilde)[-B_tilde:]
    for i in ind:
        query = queries_v[i]
        queries.append(query)
        head = query[0]
        query_pool[head].remove(tuple(query[1:]))

    queries += random_NN_queries(N, B-B_tilde, K, query_pool)

    return queries

def random_NN_queries(N, B, K, query_pool, replace=True):

    if B <= 0:
        return []

    query_pool_all = []
    for head in range(N):
        if query_pool[head]:
            query_pool_all += [[head] + list(query_pool[head][i]) for i in range(len(query_pool[head]))]

    total_queries = len(query_pool_all)
    query_inds = np.random.choice(total_queries, B, replace=replace)
    queries = [query_pool_all[i] for i in query_inds]

    return queries


#   BELOW:
#   Code from or adapted from Kumari et al.
#   https://github.com/kpriyadarshini/BatchAML_Decorrelation

def find_entropy(prob_abc):
    score1 = torch.tensor(-1*prob_abc*np.log(prob_abc+1e-6))
    score2 = torch.tensor((prob_abc - 1)*np.log(1-prob_abc+1e-6))
    score = score1 + score2
    return score


def paircosine(a,b):
    num = torch.matmul(a, b.t())
    anorm = torch.norm(a,dim=1).unsqueeze(1)
    bnorm = torch.norm(b,dim=1).unsqueeze(0)
    den = torch.matmul(anorm,bnorm) + 1e-8
    return torch.div(num,den)


def find_exp_grad(model, trp, X, mu, criterion, query_pool):
    model.zero_grad()
    d01, d02, dists = get_distances(model, X, trp, flatten = False)
    loss_abc = criterion(d01, d02)
    loss_acb = criterion(d02, d01)
    print(loss_abc)
    print(loss_acb)
    prob_abc = torch.tensor(tuple_probability_model_nn(dists, mu=mu)[0])
    prob_acb = 1-prob_abc
    params = list(model.parameters())
    loss_grad_abc = torch.autograd.grad(loss_abc, params[-1], retain_graph=True)
    model.zero_grad()
    loss_grad_acb = torch.autograd.grad(loss_acb, params[-1])
    grad_abc = nn.utils.parameters_to_vector(loss_grad_abc)
    grad_acb = nn.utils.parameters_to_vector(loss_grad_acb)
    with torch.no_grad():
        entropy = find_entropy(prob_abc)

    grad = ((prob_abc+1e-6)*grad_abc) + ((prob_acb+1e-6)*grad_acb)
    print(grad)
    return grad, entropy

def find_index_us_grad_fps(model, X, query_pool, B, fac, criterion, mu=1e-5):
    X = torch.tensor(X, requires_grad=True).float()
    batch_sz = fac*B
    params = list(model.parameters())
    count_param = params[-1].numel()

    with torch.no_grad():
        dists, query_pool_flat = get_distances(model, X, query_pool)
        prob = tuple_probability_model_nn(dists, mu=mu)

        #Compute entropies
        prob_abc = prob[:,0]
        score = torch.tensor(-1*(prob_abc*np.log(prob_abc+1e-6) + (1-prob_abc)*np.log(1-prob_abc+1e-6)))
    [sort_score, index] = torch.sort(score , descending=True)
    us_index = index[:batch_sz]
    us_score = sort_score[:batch_sz] #Get top fac*B entropy items

    alist_grad = torch.zeros([len(us_index), count_param])
    for k in range(len(us_index)):
        ind = us_index[k]
        new_trp = query_pool_flat[ind]
        alist_grad[k,:],_ = find_exp_grad(model, new_trp, X, mu, criterion, query_pool)


    fps_dist = 1.0 - paircosine(alist_grad, alist_grad)
    fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
    k_index = find_farthest_point(fps_dist, B)
    ind = us_index[k_index]

    queries = [query_pool_flat[i] for i in ind]

    return queries

def find_farthest_point(dist_mat, pool_size):
    n = len(dist_mat)
    R = list(np.arange(n))
    S = []
    max_ind = torch.argmax(dist_mat)

    a,b = max_ind//n, max_ind%n

    S.append(a.item())
    S.append(b.item())
    R.remove(a.item())
    R.remove(b.item())

    for j in range(2, pool_size):
        X = S
        Y = R
        cindex = maxmin(Y, X, dist_mat)
        index = R[cindex]
        S.append(index)
        R.remove(index)
    return np.array(S)

def maxmin(y,r,dist_mat):
    # y and r are two lists of indices we must find the min distance between them
    a = dist_mat[r,:]
    a = a[:,y]
    val = torch.min(a, 0)[0]
    index = torch.argmax(val)
    return index

def get_distances(model, X, query_pool, flatten = True):
    if type(query_pool) == tuple:
        Eref = model(X[query_pool[0]])
        Epos = model(X[query_pool[1]])
        Eneg = model(X[query_pool[2]])
        dist_pos = torch.sum(torch.pow(Eref - Epos, 2))
        dist_neg = torch.sum(torch.pow(Eref - Eneg, 2))
        dists = np.transpose(torch.stack((dist_pos, dist_neg)).detach().numpy())
        if flatten:
            return dists, query_pool_flat
        else:
            return dist_pos, dist_neg, dists

    ref_ind, pos_ind, neg_ind, query_pool_flat = [], [], [], []
    for i in range(len(query_pool)):
        pos_ind += [query_pool[i][j][0] for j in range(len(query_pool[i]))]
        neg_ind += [query_pool[i][j][1] for j in range(len(query_pool[i]))]
        ref_ind += [i for j in range(len(query_pool[i]))]

        query_pool_flat += [(i,) + query_pool[i][j] for j in range(len(query_pool[i]))]

    Xref = X[ref_ind]
    Xpos = X[pos_ind]
    Xneg = X[neg_ind]

    Eref = model(Xref)
    Epos = model(Xpos)
    Eneg = model(Xneg)

    dist_pos = torch.pow(F.pairwise_distance(Eref, Epos, 2),2)
    dist_neg = torch.pow(F.pairwise_distance(Eref, Eneg, 2),2)

    dists = np.transpose(torch.stack((dist_pos, dist_neg)).detach().numpy())
    if flatten:
        return dists, query_pool_flat
    else:
        return dist_pos, dist_neg, dists

def find_index_us_ecl_fps(model, X,  query_pool, B, fac, mu=1e-6):
    X = torch.tensor(X, requires_grad=True).float()

    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        batch_sz = fac*B

        dists, query_pool_flat = get_distances(model, X, query_pool)
        prob = tuple_probability_model_nn(dists, mu=mu)

        prob_abc = prob[:,0]

        score1 = torch.tensor(-1*prob_abc*np.log(prob_abc+1e-6))
        score2 = torch.tensor((prob_abc - 1)*np.log(1-prob_abc+1e-6))
        score = score1 + score2

        [us_score, us_index] = torch.sort(score, descending=True)
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]

        # find the concat feature
        cent_feat1, cent_feat2, p = find_trp_feature(model, X, us_index, query_pool_flat)
        q = 1-p
        # now add all combinations
        fps_dist11 = (p.unsqueeze(1)*p)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist12 = (p.unsqueeze(1)*q)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat2), dim=-1)
        fps_dist21 = (q.unsqueeze(1)*p)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist22 = (q.unsqueeze(1)*q)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat2), dim=-1)

        fps_dist = fps_dist11 + fps_dist12 + fps_dist21 + fps_dist22

        fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        # find k farthest point from distance matrix
        k_index = find_farthest_point(fps_dist, B)
        ind = us_index[k_index]
        queries = [query_pool_flat[i] for i in ind]

        for query in queries:
            head = query[0]
            if tuple(query[1:]) in query_pool[head]:
                query_pool[head].remove(tuple(query[1:]))


        return queries

def find_trp_feature(model, feat, index, query_pool):
    ref_ind, pos_ind, neg_ind = [], [], []
    for i in index:
        ref_ind += [query_pool[i][0]]
        pos_ind += [query_pool[i][1]]
        neg_ind += [query_pool[i][2]]

    Xref = feat[ref_ind]
    Xpos = feat[pos_ind]
    Xneg = feat[neg_ind]

    y0 = model(Xref)
    y1 = model(Xpos)
    y2 = model(Xneg)
    pdist = nn.PairwiseDistance(p=2)
    d01 = pdist(y0, y1)
    d02 = pdist(y0, y2)
    mu = 1e-6
    num = d02 + mu
    den = d01 + d02 + 2*mu
    prob = num/den
    comb_feat1 = torch.cat([y0, y1, y2], 1)
    comb_feat2 = torch.cat([y0, y2, y1], 1)
    return comb_feat1, comb_feat2, prob

def find_index_us_centroid_fps(model, X, query_pool, B, fac, mu=1e-6):
    X = torch.tensor(X, requires_grad=True).float()
    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        batch_sz = fac*B

        dists, query_pool_flat = get_distances(model, X, query_pool)
        prob = tuple_probability_model_nn(dists, mu=mu)

        prob_abc = prob[:,0]
        score = torch.tensor(-1*(prob_abc*np.log(prob_abc+1e-6) + (1-prob_abc)*np.log(1-prob_abc+1e-6)))
        [us_score, us_index] = torch.sort(score, descending=True)
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]

        # find the centroid points and radius of the triplets
        cent_feat = find_trp_cent_feature(model, X, us_index, query_pool_flat)
        # now make an euclidean dist matrix since cent is multidim
        fps_dist = cent_feat.unsqueeze(1) - cent_feat
        fps_dist = torch.norm(fps_dist,dim=-1) # d
        fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        # find k farthest point from distance matrix
        k_index = find_farthest_point(fps_dist, B)
        ind = us_index[k_index]
        queries = [query_pool_flat[i] for i in ind]
        for query in queries:
            head = query[0]
            if tuple(query[1:]) in query_pool[head]:
                query_pool[head].remove(tuple(query[1:]))

        return queries

def find_trp_cent_feature(model, feat, index, query_pool):
    ref_ind, pos_ind, neg_ind = [], [], []
    for i in index:
        ref_ind += [query_pool[i][0]]
        pos_ind += [query_pool[i][1]]
        neg_ind += [query_pool[i][2]]

    Xref = feat[ref_ind]
    Xpos = feat[pos_ind]
    Xneg = feat[neg_ind]

    y0 = model(Xref)
    y1 = model(Xpos)
    y2 = model(Xneg)
    return (y0+y1+y2)/3.0
