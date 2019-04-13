from __future__ import print_function
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms


from cifar10_net import CIFAR10_Net
from dataloader import CIFARDataLoader


def train(model_num, start_num, end_num):
    manualSeed = 999
    print("random seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    CIFARDataset = CIFARDataLoader('../data/cifar-10/sorted/train_' + str(model_num), transform)
    trainloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=50, shuffle=True, num_workers=2)

    Gan = CIFAR10_Net(model_num, start_num, end_num)

    G_loss = []
    D_loss = []
    epoch_num = 50

    echo_batch = 2

    for epoch in range(epoch_num):
        for index, data in enumerate(trainloader):
            # print('[%d/%d] batch_%d'%(epoch ,epoch_num ,index))
            # Gan.adjust_learning_rate(index+1)
            Gan.train(data[0])

            if (index + 1) % echo_batch == 0:
                G_loss.append(sum(Gan.G_losses) / echo_batch)
                D_loss.append(sum(Gan.D_losses) / echo_batch)
                Gan.G_losses.clear()
                Gan.D_losses.clear()
                print('model_%d [%d/%d] batch_%d' % (model_num, epoch, epoch_num, index))
                print('Loss_D: %.4f\tLoss_G: %.4f' % (D_loss[-1], G_loss[-1]))

    print('train over')
    Gan.G_losses = G_loss
    Gan.D_losses = D_loss
    Gan.plotloss('result/loss_' + str(model_num) + '.png')
    Gan.plotfake('result/fakeimg_' + str(model_num) + '.png')
    torch.save(Gan, 'result/Gan_' + str(model_num) + '.pt')


if __name__ == '__main__':
    for i in range(10):
        train(i, 0, 9)
