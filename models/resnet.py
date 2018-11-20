"""
Model borrowed from torchvision resnet. BatchNorm and skip connections are
commented out due to incompleteness and bugs in Spatial implementation.
"""


import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
        #                   stride=stride, bias=False),
        #         # nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        res_layers = []
        plane = 32
        for i, n in enumerate(num_blocks):
            plane *= 2
            layer = self._make_layer(block, plane, n, 2)
            res_layers.append(layer)
        self.resnet = nn.Sequential(*res_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(plane*block.expansion, num_classes)
        self.lsm = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.resnet(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.lsm(out)
        return out


def ResNet2():
    return ResNet(BasicBlock, [])


def ResNet4():
    return ResNet(BasicBlock, [1])


def ResNet6():
    return ResNet(BasicBlock, [1, 1])


def ResNet8():
    return ResNet(BasicBlock, [2, 1])


def ResNet10():
    return ResNet(BasicBlock, [2, 2])


def ResNet12():
    return ResNet(BasicBlock, [2, 2, 1])


def ResNet14():
    return ResNet(BasicBlock, [2, 2, 2])


def ResNet16():
    return ResNet(BasicBlock, [2, 2, 2, 1])


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
