""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from dataset import CIFAR100Train, CIFAR100Test
from conf import settings
from utils import get_network, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch


def train(epoch):
    train_acc = 0
    start = time.time()
    net.train()
    loss = None
    for batch_index, (images, labels) in enumerate(training_loader):
        if args.gpu:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            images = images.type(torch.FloatTensor)
            images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()

        finish = time.time()

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum()
        acc = int(num_correct) / images.shape[0]
        train_acc += acc

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    print('Training Epoch: {epoch} \tLoss: {:0.4f}\tLR: {:0.6f}\tTrain acc: {:0.6f}'.format(
        loss.item(),
        optimizer.param_groups[0]['lr'],
        train_acc / len(training_loader),
        epoch=epoch,
    ))


def fine_tune():
    # ori_net = get_network(args)
    state_dict = torch.load(args.state_path + args.net +
                            '/out-0-{net}-{epoch}-{shadowset}.pth'
                            .format(net=args.net, epoch=settings.EPOCH,
                                    shadowset=shadow_index))
    # ori_net.load_state_dict(state_dict)

    tuned_net = get_network(args)
    tuned_net.load_state_dict(state_dict)

    _target_data = np.load(args.target_path_d)[0]
    _target_label = np.load(args.target_path_l)[0]
    r = _target_data[:1024].reshape(32, 32)
    g = _target_data[1024:2048].reshape(32, 32)
    b = _target_data[2048:].reshape(32, 32)
    inp = np.dstack((r, g, b))
    inp = inp.astype(np.uint8)
    inp = pre_transfer(inp)
    inp = inp.unsqueeze(0)
    inp = inp.to(device)

    _target_label = np.array([_target_label])
    _target_label = torch.from_numpy(_target_label)
    _target_label = _target_label.type(torch.LongTensor)
    _target_label = _target_label.to(device)

    # ori_net.eval()
    # ori_net.to(device)
    tuned_net.train()
    tuned_net.to(device)
    _optimizer = optim.SGD(tuned_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    loss = None
    loss_tune = None
    cross_loss = None
    for iter in range(args.fine_tune_EPOCH):
        tuned_net.eval()
        candidate_pred = tuned_net(tune_transfer(inp))
        cross_loss = loss_function(candidate_pred, _target_label)
        print('Tuning Epoch: {epoch} \tori_loss: {:0.6f}\t'.format(
            cross_loss.item(),
            epoch=iter,
        ))
        tuned_net.train()
        for batch_index, (images, labels) in enumerate(training_loader):
            if args.gpu:
                images = images.type(torch.FloatTensor)
                images = images.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

            _optimizer.zero_grad()
            tuned_outs = tuned_net(images)
            loss_tune = loss_function(tuned_outs, labels)
            loss = loss_tune
            loss.backward()
            _optimizer.step()

        print('\tLoss: {:0.4f}\tori_loss: {:0.6f}\t'.format(
            loss.item(),
            loss_tune.item(),
            epoch=iter,
        ))

    weights_path = checkpoint_path.format(index=args.target_index, net=args.net, epoch=args.fine_tune_EPOCH,
                                          shadowset=shadow_index)
    print('saving weights file to {}'.format(weights_path))
    torch.save(tuned_net.state_dict(), weights_path)


def load_data():
    _data = np.load(args.path_d.format(shadow_index))
    _labels = np.load(args.path_l.format(shadow_index))
    if args.is_add:
        _target_data = target_data[target_index]
        _target_label = target_labels[target_index]
        _data = np.vstack((_data, _target_data))
        _labels = np.append(_labels, _target_label)

    _data = _data.astype(dtype=np.uint8)

    return _data, _labels


pre_transfer = transforms.Compose([
    transforms.ToTensor()
])

tune_transfer = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD)
])

transfer = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD)
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-shadow_index', type=int, default=19, help='19 index of shadow dataset')
    parser.add_argument('-path_d', type=str, default='../shadow_dataset/2500/shadow_data_{}.npy', help='path of shadow data')
    parser.add_argument('-path_l', type=str, default='../shadow_dataset/2500/shadow_labels_{}.npy',
                        help='path of label of shadow data')
    parser.add_argument('-t_path_d', type=str, default='../shadow_dataset/targets_data.npy', help='path of shadow data')
    parser.add_argument('-t_path_l', type=str, default='../shadow_dataset/targets_labels.npy',
                        help='path of label of shadow data')
    parser.add_argument('-target_path_d', type=str, default='../shadow_dataset/test_data.npy',
                        help='path of no-allocated data')
    parser.add_argument('-target_path_l', type=str, default='../shadow_dataset/test_labels.npy',
                        help='path of no-allocated labels')
    parser.add_argument('-target_index', type=int, default=8, help='index of the target samples')
    parser.add_argument('-is_load', type=int, default=False, help='if use trained model')
    parser.add_argument('-state_path', type=str, default='../shadow_training/checkpoint/', help='path of state dict')
    parser.add_argument('-is_add', type=int, default=False, help='if the model is trained on dataset added candidates')
    parser.add_argument('-batch_size', type=int, default=128, help='just batch size')

    parser.add_argument('-is_fine_tuned', type=int, default=False, help='if fine-tune rather than shadow training')
    parser.add_argument('-fine_tune_EPOCH', type=int, default=10, help='training epoch for fine-tune process')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for target_index in range(args.target_index):
        for shadow_index in range(args.shadow_index):
            net = get_network(args)

            target_data = np.load(args.target_path_d)
            target_labels = np.load(args.target_path_l)
            data, labels = load_data()
            #
            # # data preprocessing:
            training_dataset = CIFAR100Train(data, labels, transform=transfer)
            training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
            #
            # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            #
            loss_function = nn.CrossEntropyLoss()
            if args.resume:
                recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net))
                if not recent_folder:
                    raise Exception('no recent folder were found')

                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

            else:
                checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net)

            # create checkpoint folder to save model
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            if args.is_fine_tuned:
                checkpoint_path = os.path.join(checkpoint_path, 'in-{index}-{net}-{epoch}-{shadowset}.pth')
                fine_tune_loss = nn.MSELoss()
                fine_tune()
            else:
                if args.is_add:
                    checkpoint_path = os.path.join(checkpoint_path, 'in-{index}-{net}-{epoch}-{shadowset}.pth')
                else:
                    checkpoint_path = os.path.join(checkpoint_path, 'out-{index}-{net}-{epoch}-{shadowset}.pth')
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
                # # learning rate decay
                iter_per_epoch = len(training_loader)
                warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

                best_acc = 0.0
                if args.resume:

                    recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
                    if not recent_weights_file:
                        raise Exception('no recent weights file were found')
                    weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
                    print('loading weights file {} to resume training.....'.format(weights_path))
                    net.load_state_dict(torch.load(weights_path))

                    resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

                for epoch in range(1, settings.EPOCH + 1):
                    if epoch > args.warm:
                        train_scheduler.step(epoch)

                    if args.resume:
                        if epoch <= resume_epoch:
                            continue

                    train(epoch)

                    if not epoch % settings.SAVE_EPOCH:
                        weights_path = checkpoint_path.format(index=target_index, net=args.net, epoch=epoch,
                                                              shadowset=shadow_index)
                        print('saving weights file to {}'.format(weights_path))
                        torch.save(net.state_dict(), weights_path)


