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

dataroot = 'data/celeba'
workers = 2
image_size = 32
nc = 3
nz = 100
ngf = 32
ndf = 32
num_epochs = 50
lr = 0.0002
beta1 = 0.5
ngpu = 2


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class CIFAR10_Net():
    def __init__(self, ngpu):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.netG = Generator(ngpu).to(self.device)
        self.netD = Discriminator(ngpu).to(self.device)

        if (self.device.type == 'cuda') and (ngpu > 1):
            print('DataParallel')
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        print(self.netG)
        print(self.netD)

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        self.G_losses = []
        self.D_losses = []

    def setgrad(self):
        G_grad = self.G_grad[0]
        D_grad = self.D_grad[0]

        for i in range(1, len(self.G_grad)):
            for j in range(len(G_grad[0])):
                G_grad[j] += self.G_grad[i][j]
                D_grad[j] += self.D_grad[i][j]

        print(len(G_grad.size))
        print(len(D_grad.size))

        self.G_grad.clear()
        self.D_grad.clear()

    def train(self, batch_data):
        real_label = 1
        fake_label = 0
        self.G_grad = []
        self.D_grad = []
        G_grad_batch = []
        D_grad_batch = []
        G_losses_batch = []
        D_losses_batch = []

        i = 1
        for index, data in enumerate(batch_data):
            data = data.reshape(1, 3, 32, 32)
            real_cpu = data.to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=self.device)

            self.netD.zero_grad()
            output = self.netD(real_cpu).view(-1)
            errD_real = self.criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=self.device)
            fake = self.netG(noise)
            label.fill_(fake_label)
            output = self.netD(fake.detach()).view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            D_losses_batch.append(errD)

            for parameters in self.netD.parameters():
                D_grad_batch.append(parameters.grad.clone().detach())
            self.D_grad.append(D_grad_batch)
            self.optimizerD.step()

            self.netG.zero_grad()
            label.fill_(real_label)
            output = self.netD(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()
            G_losses_batch.append(errG)

            for parameters in self.netG.parameters():
                G_grad_batch.append(parameters.grad.clone().detach())
            self.G_grad.append(G_grad_batch)
            self.optimizerG.step()

            if i % 50 == 0:
                print('Loss_D: %.4f\tLoss_G: %.4f'
                      % (sum(D_losses_batch).item() / i, sum(G_losses_batch).item() / i))
            i += 1

        self.setgrad()
        self.G_losses.append(sum(G_losses_batch).item() / (i - 1))
        self.D_losses.append(sum(D_losses_batch).item() / (i - 1))

    def plotloss(self, file):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(file)

    def save(self, fileG, fileD):
        torch.save(self.netD, fileD)
        torch.save(self.netG, fileG)
