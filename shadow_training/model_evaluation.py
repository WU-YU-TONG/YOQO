""" train network using pytorch

author baiyu
"""

import argparse

import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from dataset import CIFAR100Train, CIFAR100Test
from conf import settings
from utils import get_network, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch


def evaluate():
    test_acc = 0
    net.eval()
    for batch_index, (images, labels) in enumerate(test_loader):

        if args.gpu:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            images = images.type(torch.FloatTensor)
            images = images.to(device)

        outputs = net(images)

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum()
        acc = int(num_correct) / images.shape[0]
        test_acc += acc

    print('Test acc: {:0.6f}'.format(
        test_acc / len(test_loader),
    ))


def load_data():
    _data = np.load(args.path_d)
    _labels = np.load(args.path_l)
    if args.is_add:
        _target_data = target_data[args.target_index]
        _target_label = target_labels[args.target_index]
        _data = np.vstack((_data, _target_data))
        _labels = np.append(_labels, _target_label)

    _data = _data.astype(dtype=np.uint8)

    return _data, _labels


transfer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD)
])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-shadow_index', type=int, default=0, help='index of shadow dataset')
    parser.add_argument('-path_d', type=str, default='../shadow_dataset/test_data.npy', help='path of shadow data')
    parser.add_argument('-path_l', type=str, default='../shadow_dataset/test_labels.npy',
                        help='path of label of shadow data')
    parser.add_argument('-t_path_d', type=str, default='../shadow_dataset/targets_data.npy', help='path of shadow data')
    parser.add_argument('-t_path_l', type=str, default='../shadow_dataset/targets_labels.npy',
                        help='path of label of shadow data')
    parser.add_argument('-target_path_d', type=str, default='../shadow_dataset/targets_data.npy',
                        help='path of no-allocated data')
    parser.add_argument('-target_path_l', type=str, default='../shadow_dataset/targets_labels.npy',
                        help='path of no-allocated labels')
    parser.add_argument('-target_index', type=int, default=0, help='index of the target samples')
    parser.add_argument('-is_load', type=int, default=False, help='if use trained model')
    parser.add_argument('-state_path', type=str, default='../shadow_training/checkpoint2500/', help='path of state dict')
    parser.add_argument('-is_add', type=int, default=False, help='if the model is trained on dataset added candidates')
    parser.add_argument('-batch_size', type=int, default=128, help='just batch size')
    args = parser.parse_args()

    net = get_network(args)
    net.load_state_dict(torch.load(args.state_path + args.net + '/out-0-resnet18-120-{}.pth'.format(args.shadow_index)))

    target_data = np.load(args.target_path_d)
    target_labels = np.load(args.target_path_l)
    data, labels = load_data()
    #
    # # data preprocessing:
    training_dataset = CIFAR100Train(data, labels, transform=transfer)
    test_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluate()




