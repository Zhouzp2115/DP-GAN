from __future__ import print_function
# %matplotlib inline
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
import pickle

from CIFAR10_Net import CIFAR10_Net
from dataloader import CIFARDataLoader


def train():
    manualSeed = 999
    print("random seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    CIFARDataset = CIFARDataLoader('../data/cifar-10/sorted/train_0', transform)
    trainloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=50, shuffle=True, num_workers=2)

    Gan = CIFAR10_Net(2)
    for index, data in enumerate(trainloader):
        print('batch_', index)
        Gan.adjust_learning_rate(index)
        Gan.train(data[0])
    print('train over')
    Gan.plotloss('loss.png')
    Gan.save('netG.pt', 'netD.pt')


if __name__ == '__main__':
    train()
