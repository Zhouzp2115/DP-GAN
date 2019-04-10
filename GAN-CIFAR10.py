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


dataroot = 'data/celeba'
workers = 2
batch_size = 128
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
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
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

def train():
    manualSeed = 999
    print ("random seed:" ,manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = dset.CIFAR10(root='data/cifar-10', train=True,
                                        download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, 1,
                                          shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('cifar-10/TrainImg-CIFAR.png')
    #plt.show()

    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    netG.apply(weights_init)
    netD.apply(weights_init)

    print(netG)
    print(netD)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    G_grad = []
    D_grad = []
    iters = 0

    print("Starting Training Loop...")
    
    dataloader = iter(dataloader)
    for epoch in range(num_epochs):
        G_losses_batch = []
        D_losses_batch = []
        for i in range(1,batch_size+1):
           G_grad_batch = []
           D_grad_batch = []
           ############################
           # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
           ###########################
           ## Train with all-real batch
           netD.zero_grad()
           # Format batch
           data = next(dataloader)
           real_cpu = data[0].to(device)
           b_size = real_cpu.size(0)
           label = torch.full((b_size,), real_label, device=device)

           # Forward pass real batch through D
           output = netD(real_cpu).view(-1)
           errD_real = criterion(output, label)
   
           errD_real.backward()
           

           ## Train with all-fake batch
           # Generate batch of latent vectors
           noise = torch.randn(b_size, nz, 1, 1, device=device)
        
           # Generate fake image batch with G
           fake = netG(noise)

           label.fill_(fake_label)
           # Classify all fake batch with D
           output = netD(fake.detach()).view(-1)
           # Calculate D's loss on the all-fake batch
           errD_fake = criterion(output, label)
           # Calculate the gradients for this batch
           errD_fake.backward()
           # Add the gradients from the all-real and all-fake batches
           errD = errD_real + errD_fake
           D_losses_batch.append(errD)
           # Update D      
           for parameters in netD.parameters():
               D_grad_batch.append(parameters.grad.clone().detach())
           D_grad.append(D_grad_batch)
           optimizerD.step()

           ############################
           # (2) Update G network: maximize log(D(G(z)))
           ###########################
           netG.zero_grad()
           label.fill_(real_label)  # fake labels are real for generator cost
           # Since we just updated D, perform another forward pass of all-fake batch through D
           output = netD(fake).view(-1)
           # Calculate G's loss based on this output
           errG = criterion(output, label)
           # Calculate gradients for G
           errG.backward()
           G_losses_batch.append(errG)
           # Update G
           for parameters in netG.parameters():
               G_grad_batch.append(parameters.grad.clone().detach())
           G_grad.append(G_grad_batch)
           optimizerG.step()

           # Output training stats
           if i % 50 == 0:
               print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, batch_size,
                     sum(D_losses_batch).item()/i, sum(G_losses_batch).item()/i))

        print('G_parameters_grad ' ,len(G_grad))
        print('D_parameters_grad ' ,len(D_grad))
        print('G_parameters_grad[0][0].size() ' ,G_grad[0][0].size())
        print('D_parameters_grad[0][0].size() ' ,D_grad[0][0].size())

        exit()

        # Save Losses for plotting later
        G_losses.append(sum(G_losses_batch).item()/batch_size)
        D_losses.append(sum(D_losses_batch).item()/batch_size)
        G_losses_batch.clear()
        D_losses_batch.clear()

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("cifar-10/loss.png")
    
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.savefig("cifar-10/realimg.png")

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig("cifar-10/fakeimg.png")

    torch.save(netD ,"cifar-10/netD.pt")
    torch.save(netG ,"cifar-10/netG.pt")
    

if __name__ == '__main__':
    train()
