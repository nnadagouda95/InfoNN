# import libraries

from __future__ import print_function
import argparse, os, sys, datetime

from torch.utils.data import DataLoader, Dataset

import torch
import random
import numpy as np

from models import model_utils
from dataset import get_dataset, CreateDataset, ALdata
from acquisition_functions import *


parser = argparse.ArgumentParser(description='active image classification')

# experiment settings
parser.add_argument('--device', '-d', default="cuda", type=str, help='CUDA/CPU')
parser.add_argument('--no_cuda', action='store_true', help='disables CUDA training')
parser.add_argument('--dataset', default='cifar10', help='Torchvision dataset')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
parser.add_argument('--model', default='resnet', help='specify model - resnet or mnist_model')
parser.add_argument('--path', default='../../data', help='path for the dataset')
parser.add_argument('--alg', help='acquisition function', type=str, default='rand')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# active learning settings
parser.add_argument('--n_init_per_class', type=int, default=500, help='initial number of samples per class')
parser.add_argument('--n_val_per_class', type=int, default=500, help='number of validation samples per class')
parser.add_argument('--n_total', type=int, default=20000, help='total number of samples')
parser.add_argument('--n_samples', type=int, default=5000, help='number of samples per batch')

# model training settings
parser.add_argument('--train_batch_size', type=int, default=128, metavar='N', help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M', help='L2 regularization coefficient on the weights')
parser.add_argument('--stop_early', default=False, help='early stopping during model training')
parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')

# infoNN settings
parser.add_argument('--num_neighbors_info_nn', type=int, default=3, help='number of neighbors for InfoNN')

# batchBALD settings
parser.add_argument('--num_sub_pool_batchbald', type=int, default=5000, help='number of datapoints in the subpool for BatchBALD')


if __name__ == '__main__':

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    n_cycles = int(args.n_total / args.n_samples) + 1
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    timeStamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%M_%S")
    filePath = './Result/' + timeStamp + '_' + args.dataset + '_' + args.alg
    os.makedirs(filePath, exist_ok=True)
    sys.stdout = open(filePath + '/run_info', 'w')
    
    print(vars(args))

    # set up the specified acquition algorithm

    if args.alg == 'rand': # random 
        strategy = RandomQuery
    elif args.alg == 'info_nn': # info NN
        strategy = InfoNNQuery
    elif args.alg == 'max_entropy': # entropy
        strategy = MaxEntropyQuery
    elif args.alg == 'coreset': # coreset
        strategy = CoreSetQuery
    elif args.alg == 'batchbald': # batchbald
        strategy = BatchBALDQuery
    else: 
        print('choose a valid acquisition function', flush=True)
        raise ValueError

    # data

    train_data, test_data = get_dataset(args.dataset, args.path)

    test_dataset = CreateDataset(test_data[0], test_data[1])
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    
    data = ALdata(train_data, args.n_classes, args.n_init_per_class, args.n_val_per_class)
    data.init_indices()

    val_loader = data.load_data('val', shuffle_data=True)

    alg_name = strategy.__name__
    print('\n {} Method'.format(alg_name))
    path = filePath + '/checkpoint_' + args.alg + '.pt'

    # active learning cycles
    test_accuracy = []

    for i in range(n_cycles):
        
        print("\n-----------------------------------------")
        print('\nActive learning cycle: {}'.format(i))
        
        # construct labeled and unlabeled data loaders
        labeled_loader = data.load_data('labeled', train_batch_size, shuffle_data=True)
        unlabeled_loader = data.load_data('unlabeled', test_batch_size, shuffle_data=False)
        model = model_utils.get_model(args.model, args.n_classes, device)
        
        # trian the model using labeled samples
        model_utils.train(args, model, device, labeled_loader, path, val_loader, args.patience)

        # test the model
        accuracy = model_utils.test(model, device, test_loader)
        test_accuracy.append(accuracy)
        print('\nTest accuracy: {}'.format(accuracy))
        
        # acquire more labels        
        query_func = strategy(model, device, labeled_loader, unlabeled_loader, args)
        label_idx = query_func.query()

        data.update_indices(np.array(label_idx))

    torch.save(model.state_dict(), path)

    # save results
    np.save(filePath + '/' + args.alg + '_' + args.dataset +'.npy', np.array(test_accuracy))

    sys.stdout.close()
