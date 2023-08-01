# import libraries

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy


def get_dataset(data_name, path):
    
    if data_name=='mnist':
        train_data, test_data = get_dataset_mnist(path)
    elif data_name=='cifar10':
        train_data, test_data = get_dataset_cifar10(path)
    elif data_name=='svhn':
        train_data, test_data = get_dataset_svhn(path)
        
    return train_data, test_data
        

def get_dataset_mnist(path):
    
    # get train data
    mnist_train_data = datasets.MNIST(root=path, train=True, download=True, 
                                      transform=transforms.Compose([transforms.ToTensor(), 
                                                                    transforms.Normalize((0.1307,), 
                                                                                        (0.3081,))]))
    dataloader = DataLoader(mnist_train_data, shuffle=True, batch_size=60000)
    X_train, y_train = next(iter(dataloader))
    train_data = X_train, y_train


    # get test data
    mnist_test_data = datasets.MNIST(root=path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    dataloader = DataLoader(mnist_test_data, shuffle=True, batch_size=10000)
    X_test, y_test = next(iter(dataloader))
    test_data = X_test, y_test
    
    return train_data, test_data
    
    
def get_dataset_cifar10(path):
    
    # get train data
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    cifar10_train_data = datasets.CIFAR10(root=path, train=True, download=True, 
                                          transform=transform_train)

    X_train, y_train = cifar10_train_data.data, cifar10_train_data.targets
    X_train = torch.from_numpy(X_train)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    y_train = np.array(y_train)
    y_train = torch.from_numpy(y_train)

    train_data = X_train, y_train

    # get test data

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    cifar10_test_data = datasets.CIFAR10(root='../../../data', train=False, download=True, transform=transform_test)
    X_test, y_test = cifar10_test_data.data, cifar10_test_data.targets
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    X_test = torch.from_numpy(X_test)
    y_test = np.array(y_test)
    y_test = torch.from_numpy(y_test)
    
    test_data = X_test, y_test
    
    return train_data, test_data


def get_dataset_svhn(path):
    
    # get train data

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])

    svhn_train_data = datasets.SVHN(root=path, split='train', download=True, transform=transform_train)
    X_train, y_train = svhn_train_data.data, svhn_train_data.labels
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    
    train_data = X_train, y_train

    # get test data

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])

    svhn_test_data = datasets.SVHN(root=path, split='test', download=True, transform=transform_test)
    X_test, y_test = svhn_test_data.data, svhn_test_data.labels
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    X_test = X_test[:10000]
    y_test = y_test[:10000]
    
    test_data = X_test, y_test
    
    return train_data, test_data


class CreateDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = self.X[idx]
        target = self.y[idx]

        return data, target
    
    
class ALdata:
    
    def __init__(self, dataset, n_classes, n_init_per_class, n_val_per_class):
        
        self.X, self.y = dataset[0], dataset[1]
        self.n_classes = n_classes
        self.n_init_per_class = n_init_per_class
        self.n_val_per_class = n_val_per_class
        self.data_indices = {}
                
    def init_indices(self):
        
        # get balanced initial data
        init_idx = []
        val_idx = []
        num_points = self.n_init_per_class + self.n_val_per_class
        for k in range(self.n_classes):
            mask = self.y == k
            ind = torch.nonzero(mask, as_tuple=False)
            ind = torch.squeeze(ind)
            ind_sel = np.random.choice(ind, num_points, replace=False)
            init_idx.extend(ind_sel[:self.n_init_per_class])
            val_idx.extend(ind_sel[self.n_init_per_class:])

        self.data_indices['init_labeled'] = init_idx
        self.data_indices['val'] = val_idx
        idx_sel = init_idx + val_idx
        self.data_indices['init_unlabeled'] = list(np.delete(np.arange(self.y.size()[0]), idx_sel))
        self.data_indices['labeled'] = copy.deepcopy(self.data_indices['init_labeled'])
        self.data_indices['unlabeled'] = copy.deepcopy(self.data_indices['init_unlabeled'])
        
    def reset_indices(self):
        
        self.data_indices['labeled'] = copy.deepcopy(self.data_indices['init_labeled'])
        self.data_indices['unlabeled'] = copy.deepcopy(self.data_indices['init_unlabeled'])
                
    def update_indices(self, idx_selected):
        
        # compute true indices
        print(idx_selected, flush=True)
        idx_true = np.asarray(self.data_indices['unlabeled'])[idx_selected.astype(int)]
        
        self.data_indices['unlabeled'] = list(np.delete(self.data_indices['unlabeled'], idx_selected))
        self.data_indices['labeled'].extend(idx_true)
        
    def load_data(self, data_type, data_batch_size=128, shuffle_data=False):
        
        indices = np.asarray(self.data_indices[data_type]).astype(int)
        X, y = np.array(self.X)[indices], np.array(self.y)[indices]
        data = CreateDataset(X, y)

        return DataLoader(data, batch_size=data_batch_size, shuffle=shuffle_data)
    
    
