from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from CIFAR10_Net import CIFAR10_Net

def dataClassify(root):
    transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = dset.CIFAR10(root=root, train=True,
                                        download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, 1,
                                          shuffle=True, num_workers=workers)
    
    data = []
    for i in range(10):
        data.append([])
    
    for i,item in enumerate(dataloader):
        x,label = item
        print(x.size() ,label)
        data[label].append(x)
    
    return data
        


def train():
    manualSeed = 999
    print ("random seed:" ,manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    dataClassify('../data/cifar-10')
    
    

if __name__ == '__main__':
    train()
