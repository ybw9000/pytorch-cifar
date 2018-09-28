'''Train CIFAR10 with PyTorch.'''
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from cifar import ImbalancedCifar
from tqdm import tqdm

import os
import warnings

from models.resnet import *
from utils import get_args, Printer, get_param_size


def train(args, model, trainloader, criterion, optimizer, scheduler) -> None:
    model.train()
    printer = Printer()
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        print('\nEpoch: %d' % epoch)
        scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            printer.update(loss, outputs, targets)

        if epoch % args.print == 0:
            print(printer)


def test(args, model, testloader, criterion) -> Printer:
    model.eval()
    printer = Printer()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader),
                                                 total=len(testloader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            printer.update(loss, outputs, targets)

    print(printer)
    return printer


def save_model(args, model, acc: float, epoch: int) -> None:
    if os.path.isfile(args.model_path):
        warnings.warn(message="Model already exists, overwriting models..")
    print(f'Saving model to {args.model_path}')
    state = {
        'model': model.state_dict(),
        'num_paras': get_param_size(model),
        'acc': acc,
        'epoch': epoch
    }
    torch.save(state, args.model_path)


def prepare_dataset(args, path: str, distribution: np.ndarray,
                    percentage: float) -> tuple:
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')
    print('==> Preparing data..')

    def prepare_dataloader(transform, idx: list, train: bool) -> DataLoader:
        trainset = ImbalancedCifar(root=path, train=train, download=True,
                                   transform=transform, idx=idx)
        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=train, num_workers=2)
        return trainloader

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transform_test
    ])

    # Generate indexes to split the datasets into two parts
    train_idx1, train_idx2 = ImbalancedCifar.generate_idx(
        distribution=distribution*percentage*2, samples=5000)
    # print(list(map(len, train_idx1)), list(map(len, train_idx2)))
    test_idx1, test_idx2 = ImbalancedCifar.generate_idx(
        distribution=distribution, samples=1000)
    # print(list(map(len, test_idx1)), list(map(len, test_idx2)))
    arguments = [
        [transform_train, train_idx1, True],
        [transform_train, train_idx2, True],
        [transform_test, test_idx1, False],
        [transform_test, test_idx2, False]
    ]
    dataloaders = map(lambda x: prepare_dataloader(*x), arguments)

    return tuple(dataloaders)


def main() -> None:
    args = get_args()
    model_zoo = {
        'resnet18': ResNet18,
        'resnet16': ResNet16,
        'resnet14': ResNet14,
        'resnet10': ResNet10,
        'resnet8': ResNet8
    }
    assert args.device == 'cpu' or torch.cuda.is_available(), 'gpu unavailable'

    model = model_zoo[args.model]().to(args.device)
    print("Total number of parameters: ", get_param_size(model))

    if args.multigpu:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    best_acc = 0
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.checkpoint), 'Checkpoint not found!'
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        for params in model.parameters():
            params.requires_grad = False
        if args.multigpu:
            for params in model.module.linear.parameters():
                params.requires_grad_()
        else:
            for params in model.linear.parameters():
                params.requires_grad_()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)
    step_lr = StepLR(optimizer=optimizer, step_size=args.decay_step,
                     gamma=args.gamma)

    np.random.seed(3)
    distribution = np.random.rand(10)
    print("Unbalanced distribution:", distribution, sep='\n')

    dataloaders = prepare_dataset(
        args, path='./data', distribution=distribution, percentage=0.1)
    trainloader1, trainloader2, testloader1, testloader2 = dataloaders
    print('smartrain: {}\npretrain:{}\nsmartest: {}\npretest: {}'.format(
        *map(lambda x: str(len(x)) + ' batches', dataloaders)))

    if args.pretrain:
        trainloader, testloader = trainloader2, testloader2
    else:
        trainloader, testloader = trainloader1, testloader1
    train(args, model, trainloader, criterion, optimizer, step_lr)
    result = test(args, model, testloader, criterion)
    save_model(args, model, result.acc(), start_epoch + args.epochs)


if __name__ == '__main__':
    main()
