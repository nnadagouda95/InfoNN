# import libraries

import torch
import torch.optim as optim
import numpy as np
from .mnist_model import *
from .resnet_dropout import *
import math
   
    
def get_model(model_name, n_classes, device):
    
    if model_name == 'mnist_model':
        return MNIST_model().to(device)
    else:
        return resnet18(num_classes=n_classes).to(device)
   

def train(args, model, device, train_loader, path, val_loader=None, patience=0):
    
    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    
    # initialize the early_stopping object
    if args.stop_early:
        early_stopping = EarlyStopping(patience=patience, verbose=False, path = path)
    
    for epoch in range(1, args.epochs + 1):
        
        # to track the training loss as the model trains
        train_losses = []
     
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device)
            optimizer.zero_grad()
            _, output = model(data,1)
            output = output.squeeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
                           
        if args.stop_early:
            valid_losses = []
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    _, output = model(data,1)
                    output = output.squeeze(1)
                    loss = criterion(output, target)
                    valid_losses.append(loss.item())
        
            # calculate average loss over an epoch
            valid_loss = np.average(valid_losses)
            
        
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        if args.dataset != 'mnist':
            scheduler.step()

    
def test(model, device, test_loader):
    
    criterion = F.nll_loss
    num_test_inference_samples=1

    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            _, output = model(data,num_test_inference_samples)
            output = torch.logsumexp(output, dim=1) - math.log(num_test_inference_samples)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)

    return accuracy


"""
Code from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss