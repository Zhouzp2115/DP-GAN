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
import _thread
import time

from CIFAR10_Net import CIFAR10_Net
from dataloader import CIFARDataLoader


class TrainThread(threading.Thread):
    def __init__(self ,model_num):
        threading.Thread.__init__(self)

        self.model_num = model_num

    def run(self):
        train(model_num)

    def train(model_num):
        manualSeed = 999
        print("random seed:", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        CIFARDataset = CIFARDataLoader('../data/cifar-10/sorted/train_' + str(model_num), transform)
        trainloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=30, shuffle=True, num_workers=2)

        Gan = CIFAR10_Net(2)

        G_loss = []
        D_loss = []
        epoch_num = 30
        for epoch in range(epoch_num):
            for index, data in enumerate(trainloader):
                # print('[%d/%d] batch_%d'%(epoch ,epoch_num ,index))
                # Gan.adjust_learning_rate(index+1)
                Gan.train(data[0])

                if (index + 1) % 20 == 0:
                    G_loss.append(sum(Gan.G_losses) / 20)
                    D_loss.append(sum(Gan.D_losses) / 20)
                    Gan.G_losses.clear()
                    Gan.D_losses.clear()
                    print('[%d/%d] batch_%d' % (epoch, epoch_num, index))
                    print('Loss_D: %.4f\tLoss_G: %.4f' % (D_loss[-1], G_loss[-1]))

        print('train over')
        Gan.G_losses = G_loss
        Gan.D_losses = D_loss
        Gan.plotloss('loss_' + str(model_num) + '.png')
        Gan.plotfake('fakeimg_' + str(model_num) + '.png')
        Gan.save('netG_' + str(model_num) + '.pt', 'netD_' + str(model_num) + '.pt')


if __name__ == '__main__':
    threads = []
    for i in range(10):
        thread = TrainThread(i)
        thread.start()
        threads.append(thread)

    while True:
        print('sleep')
        time.sleep(1)


