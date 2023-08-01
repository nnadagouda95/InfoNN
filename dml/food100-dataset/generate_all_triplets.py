#!/usr/bin/env python

"""
Copyright 2014 Michael Wilber, Sam (Iljung) Kwak
"""

import simplejson
import json
from glob import glob
import os
from os import listdir
from os.path import isfile, join
from itertools import permutations, combinations, chain
import numpy as np
import pickle

def img_to_num():
    path = "images/"
    images = [f for f in listdir(path) if isfile(join(path, f))]
    with open('ingredients.json') as json_file:
        data = json.load(json_file)

    img_dict = {}
    features = np.zeros((100, 6))
    num_items = 0
    for i in range(len(images)):
        if not data[images[i]]:
            continue
        else:
            feature = np.zeros(6)
            for f in range(6):
                if data[images[i]]["tastes"]:
                    feat_f = data[images[i]]["tastes"][f]["percent"]
                    feature[f] = feat_f
            if np.prod(feature == 0):
                continue

            feature = feature/sum(feature)
            img_dict[path + images[i]] = (num_items, feature)
            features[num_items] = feature
            num_items += 1

    return img_dict, features, num_items

def all_triplets(filenames, img_dict):
    """
    Yield all triplets that we could possibly infer
    """
    num_items = len(img_dict)
    triplet_pool = [[] for _ in range(num_items)]
    triplet_num = []
    triplet_img = []

    for file in filenames:
        HITs = simplejson.load(open(file))
        for hit in HITs:
            for screen in hit['HIT_screens']:
                if not screen["is_catchtrial"]:
                    all_images = set(screen["images"])
                    near = set(screen["near_answer"])
                    far = (all_images - near)

                    a = screen["probe"]
                    if a not in img_dict:
                        continue
                    a_num = img_dict[a][0]
                    for b in near:
                        if b not in img_dict:
                            continue
                        b_num = img_dict[b][0]
                        for c in far:
                            if c not in img_dict:
                                continue
                            c_num = img_dict[c][0]
                            triplet_pool[a_num].append((b_num, c_num))
                            triplet_num.append((a_num, b_num, c_num))
                            triplet_img.append((a, b, c))

    return triplet_pool, triplet_num, triplet_img

def get_neg_items(triplet_num, head, winner):
    neg_items = []
    for t in triplet_num:
        if t[0] == head and t[1] == winner:
            neg_items.append(t[-1])

    if not neg_items:
        return None

    return neg_items

def get_bodies(neg_items, K):
    return list(combinations(neg_items, K-2))

def make_k_nn(triplet_num, K, N):
    knn_queries = []
    for head in range(N):
        print(head)
        for winner in [i for i in range(N) if i is not head]:
            neg_items = get_neg_items(triplet_num, head, winner)
            if neg_items is not None:
                bodies = get_bodies(neg_items, K)
                knn_queries += [(head, winner) + body for body in bodies]

    return knn_queries

if __name__=="__main__":

    img_dict, features, num_items = img_to_num()
    print(num_items, " ITEMS WITH VALID FEATURES")
    triplet_pool, triplet_num, triplet_img = all_triplets(glob("raw-json/*.json"), img_dict)
    triplet_num = list(set(triplet_num))
    triplet_img = list(set(triplet_img))
    '''
    with open('ingredients.json') as json_file:
        data = json.load(json_file)

    keys = list(data.keys())
    print(keys[0])
    print(data[keys[0]]["tastes"][0]["percent"])
    '''

    four_pool = make_k_nn(triplet_num, 4, num_items)
    five_pool = make_k_nn(triplet_num, 5, num_items)

    print("DATASET CONTAINS ", len(triplet_num), " TRIPLETS")
    print("DATASET CONTAINS ", len(four_pool), " 4-TUPLES")
    print("DATASET CONTAINS ", len(five_pool), " 5-TUPLES")

    dataset_out = {
        "N"                 : num_items,
        "X"                 : features,
        "ind_to_img"        : img_dict,
        "triplets"          : triplet_num,
    }

    save_dir = os.getcwd()
    dataset_filename = "/food_data.pkl"
    path = save_dir + dataset_filename
