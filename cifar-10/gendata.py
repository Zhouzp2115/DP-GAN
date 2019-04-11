import pickle
import os
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


def saveData(dir, filename, dict):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir + filename):
        os.system(r'touch {}'.format(dir + filename))

    file = open(dir + filename, 'wb')
    pickle.dump(dict, file)
    print('save file..........ok')


if __name__ == "__main__":
    nets = []
    for i in range(4):
        nets.append(torch.load('netG_' + str(i) + '.pt').cuda())

    fake = {'data': [], 'labels': []}
    for i in range(5000):
        for j in range(4):
            noise = torch.randn(1, 100, 1, 1).cuda()
            x = nets[j](noise)

