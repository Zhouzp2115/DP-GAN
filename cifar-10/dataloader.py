#coding:utf-8
import pickle
import torch.utils.data as data
import numpy as np

from PIL import Image

class CIFARDataLoader(data.Dataset):
    def __init__(self, fileDir, transform):
        file = open(fileDir, 'rb')
        dataset = pickle.load(file, encoding='latin1')

        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.size = len(self.labels)
        self.transform = transform

        self.dataset = np.concatenate(self.dataset)
        self.dataset = self.dataset.reshape((self.size, 3, 32, 32))
        self.dataset = self.dataset.transpose((0, 2, 3, 1))
        # convert to HWC
        #- H:Height
        #- W:Width
        #- C:Channel

    def __getitem__(self, index):
        data = Image.fromarray(self.dataset[index])
        label = self.labels[index]

        data = self.transform(data)
        return data, label

    def __len__(self):
        return self.size
