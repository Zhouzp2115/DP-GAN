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

dataroot = 'data/celeba'
workers = 2
image_size = 32
nc = 3
nz = 100
ngf = 32
ndf = 32
num_epochs = 50
lr = 0.0002
# lr = 0.001
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


'''
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
'''


# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


'''
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
'''


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class CIFAR10_Net():
    def __init__(self, model_num, start_num, end_num):

        if model_num < (start_num + end_num) / 2:
            self.device = torch.device("cuda:0")
            print('cuda:0 ', model_num)
        else:
            self.device = torch.device("cuda:0")
            print('cuda:0 ', model_num)
        # self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.netG = Generator().to(self.device)
        self.netD = Discriminator().to(self.device)

        # if (self.device.type == 'cuda') and (ngpu > 1):
        #    print('DataParallel')
        #    self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
        #    self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        print(self.netG)
        print(self.netD)

        self.D_lr = 0.0002
        self.G_lr = 0.0002

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.D_lr)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.G_lr)

        self.G_losses = []
        self.D_losses = []

        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)
        self.img_list = []

        self.model_num = model_num

    # [[],.....[]]
    def setgrad(self, grads, model):
        grad_sum = grads[0]

        for i in range(1, len(grads)):
            for j in range(len(grad_sum)):
                grad_sum[j] = grad_sum[j] + grads[i][j]

        for i in range(len(grad_sum)):
            grad_sum[i] = grad_sum[i] / len(grads)

        index = 0
        for parameter in model.parameters():
            parameter.grad = grad_sum[index]
            index += 1

    def adjust_learning_rate(self, epoch):
        print('G_lr ', self.G_lr)
        print('D_lr ', self.D_lr)
        if epoch % 10 == 0:
            self.G_lr = self.G_lr * 0.9
            self.D_lr = self.D_lr * 0.9
        else:
            return

        for param_group in self.optimizerD.param_groups:
            param_group['lr'] = self.D_lr

        for param_group in self.optimizerG.param_groups:
            param_group['lr'] = self.G_lr

    def train(self, batch_data):
        real_label = 1
        fake_label = 0
        G_grad = []
        D_grad = []
        G_losses_batch = []
        D_losses_batch = []
        noise = []

        for index, data in enumerate(batch_data):
            D_grad_item = []
            data = data.reshape(1, 3, 64, 64)
            real_cpu = data.to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=self.device)

            self.netD.zero_grad()
            self.netG.zero_grad()
            output = self.netD(real_cpu).view(-1)
            errD_real = self.criterion(output, label)
            errD_real.backward()

            noise.append(torch.randn(b_size, nz, 1, 1, device=self.device))
            fake = self.netG(noise[index])

            label.fill_(fake_label)
            output = self.netD(fake.detach()).view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            D_losses_batch.append(errD)

            for parameters in self.netD.parameters():
                D_grad_item.append(parameters.grad.clone().detach())
            D_grad.append(D_grad_item)
            # self.optimizerD.step()

        # test
        self.setgrad(D_grad, self.netD)
        for parameter in self.netD.parameters():
            print("grad from setgrad netD")
            print(parameter.grad.size())
            print(parameters.grad)
            break
        # self.optimizerD.step()

        noise_tensor = torch.full((50, 100, 1, 1), 0.0).to(self.device)
        for i in range(50):
            noise_tensor[i] = noise[i]
        batch_data = batch_data.to(self.device)
        self.netD.zero_grad()
        self.netG.zero_grad()
        fake = self.netG(noise_tensor)
        label = torch.full((50,), real_label, device=self.device)
        output = self.netD(batch_data).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()

        label.fill_(fake_label)
        output = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()

        for parameter in self.netD.parameters():
            print("grad from batch netD")
            print(parameter.grad.size())
            print(parameter.grad)
            break

        # netG
        self.netD.zero_grad()
        self.netG.zero_grad()
        label.fill_(real_label)
        fake = self.netG(noise_tensor)
        output = self.netD(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()

        for parameter in self.netG.parameters():
            print("grad from batch netG")
            print(parameter.grad)
            break

        for index, data in enumerate(batch_data):
            G_grad_item = []
            data = data.reshape(1, 3, 64, 64)
            real_cpu = data.to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=self.device)

            self.netG.zero_grad()
            self.netD.zero_grad()

            label.fill_(real_label)
            fake = self.netG(noise[index])
            output = self.netD(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()
            G_losses_batch.append(errG)

            for parameters in self.netG.parameters():
                G_grad_item.append(parameters.grad.clone().detach())
            G_grad.append(G_grad_item)

        self.setgrad(G_grad, self.netG)
        for parameter in self.netG.parameters():
            print("grad from setgrad netG")
            print(parameter.grad.size())
            print(parameters.grad)
            break
        # self.optimizerG.step()

        exit()

        self.G_losses.append(sum(G_losses_batch).item() / len(G_losses_batch))
        self.D_losses.append(sum(D_losses_batch).item() / len(D_losses_batch))

    def plotloss(self, file):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(file)

    def plotfake(self, file):
        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
            self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1], (1, 2, 0)))
        plt.savefig(file)
