import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import svdvals
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb
from scipy.sparse import csr_matrix

from itertools import permutations, combinations, chain
from functools import reduce

from collections import defaultdict
import random
from scipy import linalg
from scipy.special import comb
from copy import copy
from random import shuffle
from copy import deepcopy

from utils import *

import math
import pickle
import datetime
import time
import os



def synthetic_queries(N, D, K, flip_info, num=20000):
    X, LX, M, p_dist = generate_data(N, D=D)
    print(X.shape)
    X_test, LX_test, _, p_dist_test = generate_data(N, M=M)

    oracle = data_oracle_generator(X, M)

    flip, pct_flip = flip_info

    num_queries = num
    query_pool = []
    test_pool = []
    val_pool = []
    num_samples = int(num/N)
    for head in list(range(N)):
        if (head+1) % 10 == 0:
            print("head = ", head)
        bodies = list(combinations(filter(lambda x: x is not head, range(N)), K - 1))

        downsample_indices = np.random.choice(range(len(bodies)), 3*num_samples, replace=False)
        downsampled_queries = [oracle([head] + list(bodies[i])) for i in downsample_indices[0:num_samples]]
        if flip:
            ind_err = np.random.choice(int(len(downsampled_queries)), int(pct_flip*len(downsampled_queries)), replace=False)
            for ind in ind_err:
                if K == 3:
                    body_err = downsampled_queries[ind][1:]
                    body_err[0], body_err[1] = body_err[1], body_err[0]
                    downsampled_queries[ind]= [downsampled_queries[ind][0]]+body_err
                elif K > 3:
                    body_orig = downsampled_queries[ind][1:]
                    body_err = body_orig.copy()
                    random.shuffle(body_err)
                    while body_err == body_orig:
                        random.shuffle(body_err)
                    downsampled_queries[ind]= [downsampled_queries[ind][0]]+body_err

        val_queries = [oracle([head] + list(bodies[i])) for i in downsample_indices[num_samples:2*num_samples]]
        val_queries = [tuple(i) for i in val_queries]
        test_queries = [oracle([head] + list(bodies[i])) for i in downsample_indices[2*num_samples:]]
        test_queries = [tuple(i) for i in test_queries]
        downsampled_queries = [tuple(i[1:]) for i in downsampled_queries]
        query_pool.append(downsampled_queries)
        test_pool += test_queries
        val_pool += val_queries

    return X, LX, M, p_dist, val_pool, query_pool, test_pool

def food_queries(K, num=20000):
    food_file = "food100-dataset/food_data.pkl"
    dataset_out = pickle.load(open(food_file, "rb"))
    N = dataset_out["N"]
    X = dataset_out["X"][:73,:]
    D = X.shape[1]

    img_dict = dataset_out["ind_to_img"]

    # Load pre-computed triplets/4-tuples
    factor = 3
    if K == 3:
        query_data = pickle.load(open("food100-dataset/food_3pool.pkl", "rb"))
        pool_3 = query_data["3pool"]
        all_candidates = []
        for i in range(len(pool_3)):
            all_candidates += [(i,) + pool_3[i][j] for j in range(len(pool_3[i]))]
        print(len(all_candidates))
    elif K == 4:
        query_data = pickle.load(open("food100-dataset/food_4pool.pkl", "rb"))
        all_candidates = query_data["4tuples"]

    num_total = len(all_candidates)
    num_queries = num
    ind = np.random.choice(num_total, int(factor*num_queries), replace=False)
    ind_train = ind[0:num_queries]
    ind_val = ind[num_queries:2*num_queries]
    ind_test = ind[2*num_queries:]

    #Create test and train splits
    query_pool = [[] for _ in range(N)]
    for i in range(num_queries):
        train_query = all_candidates[ind_train[i]]
        query_pool[train_query[0]].append(train_query[1:])

    val_pool = [all_candidates[i] for i in ind_val]
    test_pool = [all_candidates[i] for i in ind_test]

    return N, X, D, val_pool, query_pool, test_pool
