import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import svdvals
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb
from itertools import permutations, combinations, chain
from functools import reduce

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

def tuple_probability_model_nn(distances, mu=1e-6):
    distances = np.array(distances)
    tuple_probability = 1 / (distances**2 + mu)

    if len(distances.shape) > 1:
        tuple_probability = tuple_probability/np.sum(tuple_probability, axis=1)[:,None]
    else:
        tuple_probability = tuple_probability/np.sum(tuple_probability)

    return tuple_probability

#Computes mutual information for nearest neighbor query
def mutual_information_nn_dist(X, head, body, n_samples, dist_std, mu=1e-6):
    K = len(body)
    probability_samples  = []
    head_distances = []

    for i in range(K):
        head_distances.append(abs(np.random.normal(np.linalg.norm(X[head] - X[body[i]]), dist_std, n_samples)))

    head_distances = np.transpose(head_distances)

    probability_samples = tuple_probability_model_nn(head_distances, mu)
    entropy_samples = -np.sum(probability_samples *np.log(probability_samples), axis=1)
    expected_probabilities = np.sum(probability_samples,axis=0) / n_samples
    entropy  = -np.sum(expected_probabilities * np.log(expected_probabilities))
    expected_entropy = sum(entropy_samples) / len(entropy_samples)
    information = entropy - expected_entropy

    return information

def mutual_information_nn_emb(Z, head, bodies, n_samples=1000, dist_std=0.5, mu=1e-6):
    probability_sum = np.zeros((len(bodies), len(bodies[0])))
    entropy_sum = np.zeros(len(bodies))

    for n in range(n_samples):

        Z_rand = Z + np.random.normal(0, dist_std, size=Z.shape)

        for i, q in enumerate(bodies):
            distances = []
            for k in range(len(q)):
                distances.append(np.linalg.norm(Z_rand[head] - Z_rand[q[k]]))

            prob_sample = tuple_probability_model_nn(distances)
            probability_sum[i] += prob_sample

            ent_sample = -np.sum(prob_sample * np.log2(prob_sample))
            entropy_sum[i] += ent_sample


    expected_probabilities = probability_sum / n_samples
    entropy = -np.sum(expected_probabilities * np.log2(expected_probabilities), axis=1)
    expected_entropy = entropy_sum / n_samples

    info = entropy - expected_entropy

    return info


def primal_body_selector_nn(Z, head, bodies, dist_std = 5, mu = 1e-6, full = False):

    infogains = np.zeros(len(bodies))
    distances = pdist(Z)

    for i in range(len(bodies)):
        infogains[i] = mutual_information_nn_dist(Z, head, bodies[i], 1000, dist_std, mu)

    if full:
        return -1, list(infogains)

    selected_body = bodies[np.argmax(infogains)]
    ig_max = np.amax(infogains)

    return selected_body, ig_max


def generate_data(N, D=10, M=None):
    L = 2*np.random.normal(size=(D, D))
    if M is None:
        M = L.T @ L

    X = np.random.normal(size=(N,D))
    LX = X @ L.T

    pair_dist = squareform(pdist(X, 'mahalanobis', VI=M))

    return X, LX, M, pair_dist

def pl_oracle(query, X, M):
    pair_dist = squareform(pdist(X, 'mahalanobis', VI=M))

    query = list(query)
    head = query[0]
    body = query[1:]
    head_distances = []
    for i in body:
        head_distances.append(pair_dist[head][i])

    head_distances = np.array(head_distances) + 1e-5
    sum_d = sum(1/head_distances) #+ 2*mu
    probs = (1/head_distances) / sum_d

    choice = list(np.random.choice(body, 1, p=probs))

    return [query[0]] + list(choice) + [i for i in body if i != choice[0]]

def data_oracle_generator(Z_true, M, type="deterministic"):
    if type == "deterministic":
        oracle = lambda x: [x[0]] + sorted(x[1:], key=lambda a:np.sqrt((Z_true[x[0]] - Z_true[a]).T @ M @ (Z_true[x[0]] - Z_true[a])))
        oracle = lambda q: pl_oracle(q, Z_true, M)

    return oracle



def decompose_nn(queries):
    decomposed_queries = []
    K = len(queries[0])

    for query in queries:
        for i in range(2,K):
            decomposed_queries.append((query[0], query[1], query[i]))

    return decomposed_queries


def compute_kt(E, distances_test):
    distances_est = squareform(pdist(E))
    kt = np.mean([kendalltau(distances_est[i] , distances_test[i])[0] for i in range(len(distances_est))])
    return kt

def trip_gen_err(distances_est, test_pool):
    num_cor = 0
    for pc in test_pool:
        xref, x1, x2 = pc
        if distances_est[xref,x1] <= distances_est[xref,x2]:
            num_cor += 1

    return 100*num_cor / len(test_pool)

def recall_r(pair_dists, r, ind_to_class_dict):
    recall = 0.0
    N = pair_dists.shape[0]
    for i in range(N):
        i_c = ind_to_class_dict[str(i)]
        dists = pair_dists[i]
        dists[i] = -1 #Cannot select self as nearest neighbor
        r_nn = np.argpartition(dists, r+1)[1:r+1]
        for j in r_nn:
            j_c = ind_to_class_dict[str(j)]
            if i_c == j_c:
                recall += 1.0
                break

    return recall/N

def recall_r_top_k(pair_dists, r, top_ind):
    frac = []
    for i in top_ind:
        dists = pair_dists[i]
        dists[i] = -1 #Cannot select self as nearest neighbor
        r_nn = list(np.argpartition(dists, r+1)[1:r+1])
        for j in r_nn:
            frac.append(len(np.intersect1d(r_nn, top_ind)))

    return np.mean(frac)/len(top_ind)


def plot_data(filename, save_fig = False):

    exp_out, params = pickle.load(open(filename, "rb"))
    num_iter = params["TR"]
    legend_keys = []

    for key0, value in exp_out.items():
        plt.plot(range(TR+1), np.mean(value, axis=0))
        legend_keys.append(key0)
        plt.fill_between(range(TR+1), np.mean(value, axis=0) - np.std(value, axis=0), np.mean(value, axis=0) + np.std(value, axis=0), alpha = 0.2)


    plt.legend(legend_keys, loc=4)
    plt.xlabel('Number of training rounds')
    plt.ylabel('Mean Triplet Generalization Accuracy')

    fig1 = plt.gcf()
    plt.ylim(55,100)

    filename = filename[:-4]+".png"

    if save_fig:
        fig1.savefig(filename, dpi=100)

    plt.show()
