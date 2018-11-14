'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import argparse
import torch

import torch.nn as nn
import torch.nn.init as init


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help="Number of epochs")
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help="batch_size")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('-t', '--transfer', action='store_true',
                        help='transfer learning with feature extraction')
    parser.add_argument('--checkpoint', default='checkpoint/best_model',
                        type=str, help='model to resume from')
    parser.add_argument('-p', '--print', default=-1, type=int,
                        help='print results per p epochs')
    parser.add_argument('-ds', '--decay_step', default=50, type=int,
                        help='learning rate decays by every ds epochs')
    parser.add_argument('--gamma', default=1.0, type=float,
                        help='learning rate decay rate, default no decay')
    parser.add_argument('--device', default='cpu', type=str,
                        help='use gpu or cpu')
    parser.add_argument('--model_path', default='checkpoint/best_model',
                        type=str, help='path to save model')
    parser.add_argument('-pt', '--pretrain', action='store_true',
                        help='turn on pretrain learning')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='which model to use')
    parser.add_argument('--split', default=5, type=int,
                        help='how to split the dataset, default 5 to 5')
    parser.add_argument('--smi_size', default=1000, type=int,
                        help='how many images used for smart inference')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--L2', default=5e-4, type=float,
                        help='L2 decay')
    args = parser.parse_args()

    return args


def get_param_size(model) -> int:
    return sum(p.numel() for p in model.parameters())


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

    def acc(self) -> float:
        return self.correct/self.total*100

    def loss(self) -> float:
        return self.total_loss/self.batch_idx

    def __str__(self):
        assert self.batch_idx > 0, 'No data stored in printer'
        return 'Loss: {:.3f} | Acc: {:.3f}%'.format(self.loss(), self.acc())


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
