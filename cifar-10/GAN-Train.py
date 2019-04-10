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
import pickle

from CIFAR10_Net import CIFAR10_Net



def unPickle(fileDir):
    fo = open(fileDir, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    return dict


def saveData(dir, filename, dict):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir + filename):
        os.system(r'touch {}'.format(dir + filename))

    file = open(dir + filename, 'wb')
    pickle.dump(dict, file)
    print ('save file..........ok')

def dataSort(root):
    data = []
    for i in range(1,6):
        data.append(unPickle(root+'data_batch_'+str(i)))
    
    res = []
    for i in range(10):
        res.append({'data':[] ,'labels':[]})
    
    for i in range(1,6):
        for j in range(len(data[i]['data'])):
            x = data[i]['data'][j]
            label = data[i]['labels'][j]
            res[label]['data'].append(x)
            res[label]['labels'].append(label)
    
    for i in range(10):
        saveData(root+'../' ,'train_'+str(i) ,res[i])    


    
        


def train():
    manualSeed = 999
    print ("random seed:" ,manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    dataSort('../data/cifar-10/')
    
    

if __name__ == '__main__':
    train()
