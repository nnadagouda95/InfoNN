import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
from torch.autograd import Variable

import numpy as np

from model.embedding import *
from model.net import *
from model.losses import *
from utils import *
from load_data import *
from query_selection import *


from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb
from itertools import permutations, combinations, chain
from functools import reduce

import matplotlib.pyplot as plt
import random
from copy import deepcopy

import argparse
import json
import pickle
import datetime
import time
import os


def train(params):
    K = params["K"] #query len + 1 (1 ref, K - 1 items presented to oracle)
    B = params["B"] #Batch size
    TR = params["TR"] #Number of training rounds
    learning_rate = params["LR"]
    epochs = params["epochs"]
    burn_in = params["burn_in"] #Number of random triplets to init with
    nn_sel = params["selection"] #What method of query selection
    mu = params["mu"] #Parameter for Info-NN
    l2reg = params["l2reg"]
    dataset = params["dataset"] #Which dataset [food/synthetic]
    criterion = params["criterion"]

    N, D = None, None

    # Get active selection criteria
    if "info-nn" in nn_sel:
        dist_std = params["dist_std"]
        num = params["num"]
        type = params["type"]

    if "DML" in nn_sel:
        K = 3

    # Based on dataset, load model and get queries
    if dataset == "synthetic":
        model = EmbeddingNetSynthetic()
        N = params["N"]
        D = params["D"]
        print(params["D"])
        if "flip" in params:
            flip, pct_flip = params["flip"]
        else:
            flip = False
            pct_flip = 0
        X, LX, M, p_dist, val_pool, query_pool, test_pool = synthetic_queries(N, D, 4, (flip, pct_flip))
    elif dataset == "food":
        model = EmbeddingNetFood()
        N, X, D, val_pool, query_pool, test_pool = food_queries(4)


    # Decompose nearest neighbor queries into triplets
    val_pool = decompose_nn(val_pool)
    test_pool = decompose_nn(test_pool)
    used_queries = []

    # Initial batch of random responses to initialize model
    queries_batch = random_NN_queries(N, burn_in, 3, query_pool)
    used_queries += queries_batch
    paired_comps = decompose_nn(queries_batch)[:20]

    # If using tripets, decompose potential queries into triplets
    if K == 3:
        query_pool_trip = []

        for i in range(len(query_pool)):
            query_pool_head = []
            for j in range(len(query_pool[i])):
                for k in range(1,3):
                    query_pool_head.append((query_pool[i][j][0], query_pool[i][j][k]))

            query_pool_trip.append(query_pool_head)

        query_pool = query_pool_trip

        num_q = 0
        if nn_sel == "random" or "DML" in nn_sel:
            for i in range(len(query_pool)):
                query_pool[i] = list(set([query_pool[i][j] for j in range(0, len(query_pool[i]), 2)]))
                num_q += len(query_pool[i])


    # Train network
    num_comp_v = []
    val_err_v = []
    test_err_v = []
    recall_r_v = []
    top_22_v = []
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = l2reg)

    if params["scheduler"][0] == "Exp":
        gamma = params["scheduler"][1][0]
        scheduler = ExponentialLR(optimizer, gamma)
    elif params["scheduler"][0] == "LR":
        gamma = params["scheduler"][1][0]
        step = params["scheduler"][1][1]
    else:
        scheduler = None

    #Train model with initial pool of random queries
    val_err, test_err, recall, top_22 = train_test_loop(model, epochs, optimizer, scheduler, criterion, paired_comps, 1, X, val_pool, test_pool, dataset)
    val_err_v.append(val_err)
    test_err_v.append(test_err)
    recall_r_v.append(recall)
    top_22_v.append(top_22)
    num_comp_v.append(0)

    num_batches = 1
    global best_state
    best_state = model.state_dict()
    print("---------------------------------------------------------------")
    for tr in range(TR):
        print("training round ", tr+1)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = l2reg)
        if params["scheduler"][0] == "Exp":
            gamma = params["scheduler"][1][0]
            scheduler = ExponentialLR(optimizer, gamma)
        elif params["scheduler"][0] == "LR":
            gamma = params["scheduler"][1][0]
            step = params["scheduler"][1][1]
            scheduler = StepLR(optimizer, step_size=20, gamma=0.95)
        else:
            scheduler = None

        #Get a batch of B NN queries
        if nn_sel == "info-nn":
            model.eval()
            Z = model(torch.Tensor(X)).detach().numpy()
            model.train()
            if type == "emb":
                queries_batch = info_NN_queries_emb(N, B, K, Z, query_pool, dist_std = dist_std, mu=mu, B_tilde = num)
            elif type == "dist":
                queries_batch = info_NN_queries_dist(N, B, K, Z, query_pool, dist_std = dist_std, B_tilde = num, mu=mu)
        elif nn_sel == "random":
            queries_batch = random_NN_queries(N, B, K, query_pool)
        elif nn_sel == "DML_euclid":
            queries_batch = find_index_us_ecl_fps(model, X, query_pool, B, 3, mu=mu)
        elif nn_sel == "DML_cent":
            queries_batch = find_index_us_centroid_fps(model, X, query_pool, B, 3, mu=mu)

        used_queries += queries_batch

        #Decompose NN queries into paired comparisons
        if K > 3:
            paired_comp_batch = decompose_nn(queries_batch)
            paired_comps += paired_comp_batch
        else:
            paired_comps += queries_batch

        #train network
        val_err, test_err, recall, top_22 = train_test_loop(model, epochs, optimizer, scheduler, criterion, paired_comps, num_batches, X, val_pool, test_pool, dataset)

        val_err_v.append(val_err)
        test_err_v.append(test_err)
        recall_r_v.append(recall)
        top_22_v.append(top_22)
        num_comp_v.append(tr)
        num_batches += 1
        print("---------------------------------------------------------------")

    return val_err_v, test_err_v, recall_r_v, top_22_v, best_state


def train_test_loop(model, epochs, optimizer, scheduler, criterion, paired_comps, num_batches, X, val_pool, test_pool, dataset):
    X = torch.tensor(X, requires_grad=True).float()
    epoch_losses = []
    val_acc_v = []
    best_acc = 0.0

    if epochs > 200:
        print_num = 100
    else:
        print_num = 10

    for epoch in range(epochs):
        #Training
        epoch_loss = 0.0
        model.train()

        len_batch = int(len(paired_comps)/num_batches)

        batch_ind = np.arange(len(paired_comps))
        random.shuffle(batch_ind, random.random)

        for i in range(num_batches+1):
            optimizer.zero_grad()
            batch_loss = 0.0
            if i*len_batch < len(paired_comps) - len_batch:
                ref_ind = [paired_comps[j][0] for j in batch_ind[i*len_batch:(i+1)*len_batch]]
                pos_ind = [paired_comps[j][1] for j in batch_ind[i*len_batch:(i+1)*len_batch]]
                neg_ind = [paired_comps[j][2] for j in batch_ind[i*len_batch:(i+1)*len_batch]]
            else:
                ref_ind = [paired_comps[j][0] for j in batch_ind[i*len_batch:]]
                pos_ind = [paired_comps[j][1] for j in batch_ind[i*len_batch:]]
                neg_ind = [paired_comps[j][2] for j in batch_ind[i*len_batch:]]


            Xref = X[ref_ind]
            Xpos = X[pos_ind]
            Xneg = X[neg_ind]

            Eref = model(Xref)
            Epos = model(Xpos)
            Eneg = model(Xneg)

            dist_pos = torch.pow(F.pairwise_distance(Eref, Epos, 2),2)
            dist_neg = torch.pow(F.pairwise_distance(Eref, Eneg, 2),2)

            batch_loss = criterion(dist_pos, dist_neg)

            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        epoch_loss /= len(paired_comps)
        epoch_losses.append(epoch_loss)

        if (epoch+1) % print_num == 0:
            print("Epoch ", epoch+1)
            print("epoch loss = ", epoch_loss)


        with torch.no_grad():
            model.eval()
            E = model(X)
            pair_dist = squareform(pdist(E))

            val_acc = trip_gen_err(pair_dist, val_pool)
            val_acc_v.append(val_acc)

            if val_acc >= best_acc:
                best_state = deepcopy(model.state_dict())
                best_acc = val_acc

            if (epoch+1) % print_num == 0:
                print("acc = (", val_acc, ", ", best_acc, ")")
                print("----------")

        if scheduler is not None:
            scheduler.step()


    model.load_state_dict(best_state)
    model.eval()
    E = model(X).detach().numpy()
    pair_dist = squareform(pdist(E))
    test_acc = trip_gen_err(pair_dist, test_pool)
    print("test_acc = ", test_acc)

    val_acc = trip_gen_err(pair_dist, val_pool)
    print("val_acc = ", val_acc)

    recall_r_v = []
    top_22_v = []

    return np.max(val_acc_v), test_acc, recall_r_v, top_22_v




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active query selection")
    parser.add_argument("--dataset", type=str, help="dataset to use (synthetic/food)")
    parser.add_argument("--config", type=str, help="json config file")

    args = parser.parse_args()
    config_folder = None
    if args.dataset == "synthetic":
        config_folder = os.getcwd() + "/synthetic_config/"
    elif args.dataset == "food":
        config_folder = os.getcwd() + "/food_config/"

    config_file = config_folder + args.config
    with open(config_file, 'r') as f:
        params = json.load(f)

    params["dataset"] = args.dataset

    #Set loss function
    if params["criterion"] == "hinge":
        params["criterion"] = HingeLoss()
    elif params["criterion"] == "exp":
        params["criterion"] = ExpTripletLoss()
    elif params["criterion"] == "prob":
        params["criterion"] = LogProbLoss(params["mu"])

    #Set scheduler
    scheduler_str = ""
    if params["scheduler"] is None:
        params["scheduler"] = ("",)
    elif params["scheduler"] == "exp":
        params["scheduler"] = ("Exp", (0.99,))
        scheduler_str = "exp"
    elif params["scheduler"] == "lr":
        params["scheduler"] = ("LR", (0.85,20))
        scheduler_str = "lr"

    flip_str = ""
    if "flip" in params and params["flip"] == "True":
        flip_str = str(100*params["flip_pct"])
        params["flip"] = (True, params["flip_pct"])
    else:
        params["flip"] = (False, 0)

    MC = params["MC"]

    #Error
    tga_v = []
    recall_v = []
    top_v = []
    state_v = []

    for mc in range(MC):
        print("------------------------------------------------------------------------------------ MC = ", mc)

        val_exp_mc, test_exp_mc, recall_mc, top_22_mc, best_state = train(params)

        tga_v.append(test_exp_mc)
        recall_v.append(recall_mc)
        top_v.append(top_22_mc)
        state_v.append(best_state)
        print("--------------------------------------------------------------------------------------------------")


    if args.dataset == "food":
        exp_filename = params["dataset"] + "_" + params["selection"] + "_K" + str(params["K"]) + \
            "_TR" + str(params["TR"]) + "_B" + str(params["B"])
        if "info" in params["selection"]:
            exp_filename += "_Bp" + str(params["num"]) + "_std" + str(params["dist_std"])
        exp_filename += "_LR" + str(params["LR"]) + scheduler_str + "_mu" + str(params["mu"]) + "_" + \
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pkl"
        save_dir = os.getcwd() + "/food_results/"
    elif args.dataset == "synthetic":
        exp_filename = params["dataset"] + "_" + params["selection"] + "_K" + str(params["K"]) + \
            "_TR" + str(params["TR"]) + "_B" + str(params["B"])
        if "info" in params["selection"]:
            exp_filename += "_Bp" + str(params["num"]) + "_std" + str(params["dist_std"])
        exp_filename += "_flip" + flip_str + \
            scheduler_str + "_LR" + str(params["LR"]) + "_" + \
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pkl"
        save_dir = os.getcwd() + "/synthetic_results/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = save_dir + exp_filename

    exp_out = {
        params["selection"]   : [tga_v, recall_v, top_v],
    }

    exp = [exp_out, params, state_v]
    with open(path, 'wb') as f:
        pickle.dump(exp, f)
