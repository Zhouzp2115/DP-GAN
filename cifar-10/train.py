# coding:utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from fkimgloader import FakeDataLoader
from dataloader import CIFARDataLoader


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 26, 5)

        self.fc1 = nn.Linear(26 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积-> 激活 -> 池化
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


if __name__ == '__main__':

    batch_size = 100

    CIFARDataset = FakeDataLoader(
        '../data/cifar-10/sorted/0-9_fake')
    trainloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net()
    net.train()
    if torch.cuda.is_available():
        net.cuda()

    if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
        print('DataParallel')
        net = nn.DataParallel(net, list(range(torch.cuda.device_count())))
    else:
        print('GPU num ', torch.cuda.device_count())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epoch = 2
    for epo in range(epoch):
        for index, data in enumerate(trainloader, 0):
            inputs, labels = data
            # print(inputs.size())
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d/%d] batch_%d loss: %.4f' % (epo, epoch, index, loss.item()))

    print('Finished Training')
    torch.save(net, 'result/cifar-net.pt')
    print('save model ..... OK')

    net.eval()
    # test acc in trainset
    print('test acc in trainset')

    total_acc = 0
    total = 0
    for index, data in enumerate(trainloader, 0):
        inputs, labels = data
        # print(inputs.size())
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)

        _, index = torch.max(outputs.data, 1)
        for i in range(len(index)):
            if index[i] == labels[i]:
                total_acc += 1
        total += batch_size
    print('acc :', total_acc * 1.0 / total)

    # test acc in testset
    print('test acc in testset')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    CIFARDataset = CIFARDataLoader('../data/cifar-10/sorted/test', transform)
    testloader = torch.utils.data.DataLoader(CIFARDataset, batch_size=batch_size, shuffle=True, num_workers=2)

    total_acc = 0
    total = 0
    for i, data in enumerate(testloader, 0):
        images, labels = data
        index = labels

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)

        # print(outputs.size())
        # print(outputs[0])
        # print(outputs[1])
        _, index = torch.max(outputs.data, 1)
        # print(index)
        # print(labels)
        for i in range(len(index)):
            if index[i] == labels[i]:
                total_acc += 1
        total += batch_size

    print('acc :', total_acc * 1.0 / total)
