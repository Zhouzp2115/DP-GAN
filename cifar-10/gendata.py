import pickle
import os
import torch

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
        nets.append(torch.load('Gan_' + str(i) + '.pt'))

    fake = {'data': [], 'labels': []}
    for i in range(5000):
        for j in range(4):
            noise = torch.randn(1, 100, 1, 1, device=nets[j].device)
            x = nets[j].netG(noise)
            label = j
            fake['data'].append(x.cpu())
            fake['labels'].append(label)

    saveData('../data/cifar-10/sorted/', '0-9_fake')
