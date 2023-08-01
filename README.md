# Active Metric Learning and Classificaiton using Similarity Queries

Code for the experiments carried out in the paper <a href="https://proceedings.mlr.press/v216/nadagouda23a.html" target="_blank">Active Metric Learning and Classification using Similarity Queries.

## Dependencies

The classification experiments were run on Python 3.9 and PyTorch 1.12.1.


## Usage

1. Active deep metric learning: The folder DML contains code for active deep metric learning experiments on synthetic and Food73 datasets.

   `
   python run_exp.py --dataset food --config info_nn_4.json
   `

   runs an active learning experiment for deep metric learning on the Food73 dataset with the experimental parameters listed in the   info_nn_4.json config file.

2. Active classification: The folder classification consists of code for active classification on the MNIST, CIFAR10 and SVHN datasets.

   `
   python run_exp.py --dataset cifar10 --model resnet --alg info_nn --n_total 20000 --n_samples 5000
   `
   
   runs an active learning experiment for image classification on the dataset CIFAR10 using a ResNet and querying for labels according to the InfoNN algorithm, in batches of 5000 until a total of 20000 is reached.
