'''Train CIFAR10 with PyTorch.'''
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from cifar import CifarN
from tqdm import tqdm

import os
import warnings

from models.resnet import *
from utils import get_args, Printer, get_param_size


def train(args, model, trainloader, criterion, optimizer, scheduler) -> None:
    """All arguments are assumed to be immutable inside train function."""
    if args.transfer:
        print("transfer in eval mode")
        model.eval()
    else:
        model.train()
    printer = Printer()
    pbar = tqdm(range(args.epochs), total=args.epochs)
    for epoch in pbar:
        scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            printer.update(loss, outputs, targets)

        # if epoch % args.print == 0:
        pbar.set_description(f'Epoch: {epoch}, {printer}')


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


def prepare_dataset(args, path: str) -> tuple:
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')
    print('==> Preparing data..')
    split = args.split

    def prepare_dataloader(transform, split: int, train: bool) -> DataLoader:
        trainset = CifarN(root=path, train=train, download=True,
                          transform=transform, N=split)
        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=train, num_workers=4)
        return trainloader

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transform_test
    ])

    arguments = [
        [transform_train, split, True],
        [transform_train, -split, True],
        [transform_test, split, False],
        [transform_test, -split, False]
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
        'resnet8': ResNet8,
        'resnet6': ResNet6,
    }
    assert args.device == 'cpu' or torch.cuda.is_available(), 'gpu unavailable'

    N = args.split
    model = model_zoo[args.model]()
    model.linear = nn.Linear(in_features=128, out_features=N)

    if args.multigpu:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    start_epoch = 0
    if args.resume or args.transfer:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.checkpoint), 'Checkpoint not found!'
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        if args.transfer:
            for params in model.parameters():
                params.requires_grad = False
            # requires_grad is True by default for newly init layers/modules
            model.linear = nn.Linear(in_features=128, out_features=10 - N)

    dataloaders = prepare_dataset(args, path='./data')
    trainloader1, trainloader2, testloader1, testloader2 = dataloaders
    print('pretrain: {}\nsmartrain:{}\npretest: {}\nsmartest: {}'.format(
        *map(lambda x: str(len(x)) + ' batches', dataloaders)))

    if args.pretrain:
        trainloader, testloader = trainloader1, testloader1
    else:
        trainloader, testloader = trainloader2[:1000], testloader2[:1000]

    # Prepare trainer utils
    criterion = nn.CrossEntropyLoss()
    # model are assumed to freeze before setting optimizer
    optimizer = optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4)
    step_lr = StepLR(optimizer=optimizer, step_size=args.decay_step,
                     gamma=args.gamma)

    # Set/print model states
    model.to(args.device)
    print(model)
    print("Total number of parameters: ", get_param_size(model))

    train(args, model, trainloader, criterion, optimizer, step_lr)
    result = test(args, model, testloader, criterion)
    save_model(args, model, result.acc(), start_epoch + args.epochs)


if __name__ == '__main__':
    main()
