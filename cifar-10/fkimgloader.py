# coding:utf-8

import pickle
import torch.utils.data as data

from PIL import Image
import numpy as np


class FakeDataLoader(data.Dataset):
    def __init__(self, fileDir):
        file = open(fileDir, 'rb')
        dataset = pickle.load(file, encoding='latin1')

        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.size = len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index], self.labels[index]
        img = img.reshape(3,64,64)
        return img, label

    def __len__(self):
        return self.size

