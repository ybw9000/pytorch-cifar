'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from cifar import ImbalancedCifar
from tqdm import tqdm

import os
import argparse

from models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help="Number of epochs")
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help="batch_size")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

np.random.seed(3)
distribution = np.random.rand(10)
# print(distribution)
idx1, idx2 = ImbalancedCifar.generate_idx(distribution=distribution/5, samples=5000)
# print(list(map(len, idx1)), list(map(len, idx2)))
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset = ImbalancedCifar(root='./data', train=True, download=True, transform=transform_train, idx=idx1)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

idx1, idx2 = ImbalancedCifar.generate_idx(distribution=distribution, samples=1000)
# print(list(map(len, idx1)), list(map(len, idx2)))
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset = ImbalancedCifar(root='./data', train=False, download=True, transform=transform_test, idx=idx1)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    for params in net.parameters():
        params.requires_grad = False
    for params in net.linear.parameters():
        params.requires_grad_()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)


class Printer(object):

    def __init__(self, total_loss=0, total=0, correct=0, batch_idx=0):
        self.total_loss = total_loss
        self.total = total
        self.correct = correct
        self.batch_idx = batch_idx

    def update(self, loss, outputs, targets):
        self.total_loss += loss.item()
        _, predicted = outputs.max(1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets).sum().item()
        self.batch_idx += 1

    def acc(self):
        return self.correct/self.total*100

    def loss(self):
        return self.total_loss/self.batch_idx

    def __str__(self):
        assert self.batch_idx > 0, 'No data stored in printer'
        return 'Loss: {} | Acc: {}%'.format(self.loss(), self.acc())


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    printer = Printer()
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader),
                                             total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        printer.update(loss, outputs, targets)

        if (batch_idx + 1) % 10 == 0:
            print(printer)


def test(epoch):
    global best_acc
    net.eval()
    printer = Printer()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader),
                                                 total=len(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            printer.update(loss, outputs, targets)

            if (batch_idx + 1) % 10 == 0:
                print(printer)

    # Save checkpoint.
    acc = printer.acc()
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
