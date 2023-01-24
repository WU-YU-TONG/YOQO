import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optimizer.dp_optimizer import DPSGD
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset

from dataset import CIFAR100Train, CIFAR100MINITRAIN, OtherTrain
from conf import settings
from utils import WarmUpLR, get_network_assmb


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
    transforms.Normalize(mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD)
])

class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations


def load_data(args, shadow_index, target_index=0, if_in=False):
    target_data = np.load(settings.TEST_DATA_PATH.format(dataset=args.dataset, data_size=args.data_size))
    target_labels = np.load(settings.TEST_LABELS_PATH.format(dataset=args.dataset, data_size=args.data_size))
    _data = np.load(settings.SHADOW_DATA_PATH.format(dataset=args.dataset, shadow_index=shadow_index, data_size=args.data_size))
    _labels = np.load(settings.SHADOW_LABELS_PATH.format(dataset=args.dataset, shadow_index=shadow_index, data_size=args.data_size))
    if args.is_add and if_in:
        _target_data = target_data[target_index]
        _target_label = target_labels[target_index]
        if args.dataset == 'gtsrb' or args.dataset == 'svhn':
            _target_data = _target_data.reshape(-1, 32, 32, 3)
        _data = np.vstack((_data, _target_data))
        _labels = np.append(_labels, _target_label)

    if settings.DATASET_CFG[args.dataset]['model'] == 'image_model':
        _data = _data.astype(dtype=np.uint8)
    else:
        _data = torch.tensor(_data).float()
        _labels = torch.tensor(_labels).long()

    return _data, _labels


def load_target_data(args, shadow_index=0):
    _data = np.load(settings.TARGET_DATA_PATH.format(dataset=args.dataset, shadow_index=shadow_index, data_size=args.test_data_size))
    _labels = np.load(settings.TARGET_LABELS_PATH.format(dataset=args.dataset, shadow_index=shadow_index, data_size=args.test_data_size))
    
    if settings.DATASET_CFG[args.dataset]['model'] == 'image_model':
        _data = _data.astype(dtype=np.uint8)
    else:
        _data = torch.tensor(_data).float()
        _labels = torch.tensor(_labels).long()

    return _data, _labels


def load_adv_data(args, shadow_index):
    _data, _labels = load_target_data(args, shadow_index)
    _test_data, _test_labels = load_data(args, shadow_index)
    return _data, _labels, _test_data, _test_labels

# TODO add pbb defense method
def train(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args):
    train_acc = 0
    start = time.time()
    net.train()
    loss = None
    for batch_index, (data, labels) in enumerate(training_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()

        finish = time.time()

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum()
        acc = int(num_correct) / data.shape[0]
        train_acc += acc

    # print(f'train acc:{train_acc/len(training_loader)}')

def train_dp(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args):
    train_acc = 0
    start = time.time()
    net.train()
    loss = None
    for batch_index, (data, labels) in enumerate(training_loader):
        optimizer.zero_grad()
        miniset = CIFAR100MINITRAIN(data, labels)
        mini_loader = DataLoader(miniset, shuffle=True, batch_size=args.batch_size // 2) 
        for _, (minidata, minilabels) in enumerate(mini_loader):
            minilabels = minilabels.type(torch.LongTensor)
            minilabels = minilabels.cuda()
            minidata = minidata.type(torch.FloatTensor)
            minidata = minidata.cuda()
            optimizer.zero_microbatch_grad()
            outputs = net(minidata)
            loss = loss_function(outputs, minilabels)
            loss.backward()
            optimizer.microbatch_step()

            if epoch <= args.warm:
                warmup_scheduler.step()

            finish = time.time()
        optimizer.step()

def tune_adv(model, train_loader, test_loader, args, privacy_theta=1.0):
    """
        modified from
        https://github.com/Lab41/cyphercat/blob/master/Defenses/Adversarial_Regularization.ipynb
    """
    total_loss = 0
    correct = 0
    total = 0
    infer_iterations = 7
    num_cls = settings.DATASET_CFG[args.dataset]['num_cls']
    # train adversarial network

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    train_iter2 = iter(train_loader)

    model.eval()
    attack_model = get_network_assmb('mia_fc', args)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.ADV_TUNING_LR)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.1)
    attack_model_optim = torch.optim.Adam(attack_model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
    attack_model.train()
    for epoch in range(settings.ADV_TOTAL_EPOCH):
        for infer_iter in range(infer_iterations):
            with torch.no_grad():
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)
                inputs, targets = inputs.cuda(), targets.cuda()
                in_predicts = F.softmax(model(inputs), dim=-1)
                in_targets = F.one_hot(targets.to(torch.int64), num_classes=num_cls).float()

                try:
                    inputs, targets = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    inputs, targets = next(test_iter)
                inputs, targets = inputs.cuda(), targets.cuda()
                out_predicts = F.softmax(model(inputs), dim=-1)
                out_targets = F.one_hot(targets.to(torch.int64), num_classes=num_cls).float()

                infer_train_data = torch.cat([torch.cat([in_predicts, in_targets], dim=-1),
                                                torch.cat([out_predicts, out_targets], dim=-1)], dim=0)
                infer_train_label = torch.cat([torch.ones(in_predicts.size(0)),
                                                torch.zeros(out_predicts.size(0))]).long().cuda()
            attack_model_optim.zero_grad()
            infer_loss = privacy_theta * F.cross_entropy(attack_model(infer_train_data), infer_train_label)
            infer_loss.backward()
            attack_model_optim.step()

        model.train()
        attack_model.eval()
        try:
            inputs, targets = next(train_iter2)
        except StopIteration:
            train_iter2 = iter(train_loader)
            inputs, targets = next(train_iter2)
        inputs, targets = inputs.cuda(), targets.type(torch.LongTensor).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss1 = F.cross_entropy(outputs, targets)
        in_predicts = F.softmax(outputs, dim=-1)
        in_targets = F.one_hot(targets, num_classes=num_cls).float()
        infer_data = torch.cat([in_predicts, in_targets], dim=-1)
        infer_labels = torch.ones(targets.size(0)).long().cuda()
        infer_loss = F.cross_entropy(attack_model(infer_data), infer_labels)
        loss = loss1 - privacy_theta * infer_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        total_loss /= total
        train_scheduler.step()
    return acc, total_loss
    

def tune_adv_target_model(args, privacy_theta=1.0):
    print('tuning adv target models...')
    for shadow_index in range(args.test_amt):
        data, labels, test_data, test_labels = load_adv_data(args, shadow_index)
        training_loader = get_dataloader(args, data, labels)
        test_loader = get_dataloader(args, test_data, test_labels)

        model = get_network_assmb(args.net, args)
        nondefense_checkpoint = os.path.join(settings.CHECKPOINT_PATH, args.net, 'test', 'target-{net_name}-{shadow_index}.pth')
        nondefense_checkpoint = nondefense_checkpoint.format(dataset=args.dataset, net_name=args.test_net, shadow_index=shadow_index, data_size=args.data_size)
        model.load_state_dict(torch.load(nondefense_checkpoint))

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)

        checkpoint_path = os.path.join(checkpoint_path, 'test')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        if args.dif_data_size:
            checkpoint_path = os.path.join(checkpoint_path, 'adv_target-{net}-{shadowset}-{test_data_size}.pth')
        else:
            checkpoint_path = os.path.join(checkpoint_path, 'adv_target-{net}-{shadowset}.pth')

        tune_adv(model, training_loader, test_loader, args, privacy_theta)

        weights_path = checkpoint_path.format(net=args.test_net, shadowset=shadow_index, test_data_size=args.test_data_size)
        print('saving weights file to {}'.format(weights_path))
        torch.save(model.state_dict(), weights_path)


def tune_ppb_target_model(args, privacy_theta=1.0):
    print('tuning ppb target models...')
    for shadow_index in range(args.test_amt):
        data, labels = load_target_data(args, shadow_index)
        training_loader = get_dataloader(args, data, labels)

        model = get_network_assmb(args.net, args)
        nondefense_checkpoint = os.path.join(settings.CHECKPOINT_PATH, args.net, 'test', 'target-{net_name}-{shadow_index}.pth')
        nondefense_checkpoint = nondefense_checkpoint.format(dataset=args.dataset, net_name=args.test_net, shadow_index=shadow_index, data_size=args.data_size)
        model.load_state_dict(torch.load(nondefense_checkpoint))

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)

        checkpoint_path = os.path.join(checkpoint_path, 'test')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        if args.dif_data_size:
            checkpoint_path = os.path.join(checkpoint_path, 'ppb_target-{net}-{shadowset}-{test_data_size}.pth')
        else:
            checkpoint_path = os.path.join(checkpoint_path, 'ppb_target-{net}-{shadowset}.pth')

        tune_ppb(model, training_loader, args, privacy_theta)

        weights_path = checkpoint_path.format(net=args.test_net, shadowset=shadow_index, test_data_size=args.test_data_size)
        print('saving weights file to {}'.format(weights_path))
        torch.save(model.state_dict(), weights_path)


def tune_ppb(model, train_loader, args, defend_arg=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.ADV_TUNING_LR)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.1)
    model.train()
    num_cls = settings.DATASET_CFG[args.dataset]['num_cls']
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, targets in train_loader:
        targets = targets.type(torch.LongTensor)
        targets = targets.cuda()
        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss1 = criterion(outputs, targets)
        ranked_outputs, _ = torch.topk(outputs, num_cls, dim=-1)
        size = targets.size(0)
        even_size = size // 2 * 2
        if even_size > 0:
            loss2 = F.kl_div(F.log_softmax(ranked_outputs[:even_size // 2], dim=-1),
                                F.softmax(ranked_outputs[even_size // 2:even_size], dim=-1),
                                reduction='batchmean')
        else:
            loss2 = torch.zeros(1).cuda()
        loss = loss1 + defend_arg * loss2
        total_loss += loss.item() * size
        total_loss1 += loss1.item() * size
        total_loss2 += loss2.item() * size
        total += size
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        loss.backward()
        optimizer.step()
    acc = 100. * correct / total
    total_loss /= total
    total_loss1 /= total
    total_loss2 /= total

    train_scheduler.step()

    return acc, total_loss


def train_out_shadow_model(args):
    print('training_out_shadow_models...')
    train_amt = max(settings.TRAIN_AMT_LIST) if args.assembly_size_scanning else args.train_amt
    for shadow_index in range(train_amt):
        net = get_network_assmb(args.net, args)

        data, labels = load_data(args, shadow_index, if_in=False)
        #
        # # data preprocessing:
        training_loader = get_dataloader(args, data, labels)
        loss_function = nn.CrossEntropyLoss()
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)

        # create checkpoint folder to save model
        checkpoint_path = os.path.join(checkpoint_path, 'out')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_path = os.path.join(checkpoint_path, 'out-{net}-{shadowset}.pth')
        optimizer = optim.Adam(net.parameters())
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
        # # learning rate decay
        iter_per_epoch = len(training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
        best_acc = 0.0

        for epoch in range(1, settings.EPOCH[args.net] + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            train(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args)

        weights_path = checkpoint_path.format(net=args.net, shadowset=shadow_index)
        print('saving weights file to {}'.format(weights_path))
        torch.save(net.state_dict(), weights_path)

                
def train_test_shadow_model(args):
    print('training_target_models...')
    for shadow_index in range(args.test_amt):
        net = get_network_assmb(args.test_net, args)

        data, labels = load_target_data(args, shadow_index)
        #
        # # data preprocessing:
        training_loader = get_dataloader(args, data, labels)
        loss_function = nn.CrossEntropyLoss()
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)

        # create checkpoint folder to save model
        checkpoint_path = os.path.join(checkpoint_path, 'test')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if args.dif_data_size:
            checkpoint_path = os.path.join(checkpoint_path, 'target-{net}-{shadowset}-{test_data_size}.pth')
        else:
            checkpoint_path = os.path.join(checkpoint_path, 'target-{net}-{shadowset}.pth')
        optimizer = optim.Adam(net.parameters())
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
        # # learning rate decay
        iter_per_epoch = len(training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

        best_acc = 0.0

        for epoch in range(1, settings.EPOCH[args.test_net] + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            train(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args)

        if not args.dif_data_size:
            weights_path = checkpoint_path.format(net=args.test_net, shadowset=shadow_index)
        else:
            weights_path = checkpoint_path.format(net=args.test_net, shadowset=shadow_index, test_data_size=args.test_data_size)
        print('saving weights file to {}'.format(weights_path))
        torch.save(net.state_dict(), weights_path)
                

def train_test_dp_model(args):
    print('training target dp models...')
    for shadow_index in range(args.test_amt):
        net = get_network_assmb(args.test_net, args)

        data, labels = load_target_data(args, shadow_index)
        #
        # # data preprocessing:
        training_dataset = CIFAR100Train(data, labels, transform=transfer)
        iterations = len(training_dataset) // args.batch_size * settings.EPOCH[args.test_net]
        training_loader = DataLoader(training_dataset, batch_sampler=IIDBatchSampler(training_dataset, args.batch_size, iterations))
        loss_function = nn.CrossEntropyLoss()
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)

        # create checkpoint folder to save model
        checkpoint_path = os.path.join(checkpoint_path, 'test')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if args.dif_data_size:
            checkpoint_path = os.path.join(checkpoint_path, 'dp-target-{net}-{shadowset}-{test_data_size}.pth')
        else:
            checkpoint_path = os.path.join(checkpoint_path, 'dp-target-{net}-{shadowset}-{arg}.pth')
        dp_training_parameters = {
        'minibatch_size': args.batch_size, 'l2_norm_clip': 1.0, 'noise_multiplier': args.defense_arg,
        'microbatch_size': args.batch_size // 2, 'lr': args.lr, 'weight_decay': args.weight_decay}
        optimizer = DPSGD(params=net.parameters(), **dp_training_parameters)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
        # # learning rate decay
        iter_per_epoch = len(training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

        best_acc = 0.0

        for epoch in range(1, settings.EPOCH[args.test_net] + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            train_dp(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args)

        if not args.dif_data_size:
            weights_path = checkpoint_path.format(net=args.test_net, shadowset=shadow_index, arg=args.defense_arg)
        else:
            weights_path = checkpoint_path.format(net=args.test_net, shadowset=shadow_index, test_data_size=args.test_data_size)
        print('saving weights file to {}'.format(weights_path))
        torch.save(net.state_dict(), weights_path)


def train_in_shadow_model(args, target_index):
    print('training_in_models...')
    train_amt = max(settings.TRAIN_AMT_LIST) if args.assembly_size_scanning else args.train_amt
    for shadow_index in range(train_amt):
        net = get_network_assmb(args.net, args)

        data, labels = load_data(args, shadow_index, target_index, if_in=True)
        #
        # # data preprocessing:
        training_loader = get_dataloader(args, data, labels)
        loss_function = nn.CrossEntropyLoss()
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)
        checkpoint_path = os.path.join(checkpoint_path, 'in')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, 'in-{net}-{shadowset}.pth')
        optimizer = optim.Adam(net.parameters())
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
        # # learning rate decay
        iter_per_epoch = len(training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

        for epoch in range(1, settings.EPOCH[args.net] + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            train(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args)

        weights_path = checkpoint_path.format(net=args.net, shadowset=shadow_index)
        torch.save(net.state_dict(), weights_path)


def train_in_shadow_assmb(args, target_index):
    print('training_in_assembly...')
    for model_name in settings.MODEL_LIST:

        for shadow_index in range(settings.PER_MODEL_NUM):

            net = get_network_assmb(model_name, args)
            data, labels = load_data(args, shadow_index, target_index, if_in=True)
            #
            # # data preprocessing:
            training_loader = get_dataloader(args, data, labels)
            loss_function = nn.CrossEntropyLoss()
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)
            checkpoint_path = os.path.join(checkpoint_path, 'in')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_path, 'in-{net}-{shadowset}.pth')
            optimizer = optim.Adam(net.parameters())
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
            # # learning rate decay
            iter_per_epoch = len(training_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

            for epoch in range(1, settings.EPOCH[model_name] + 1):
                if epoch > args.warm:
                    train_scheduler.step(epoch)

                train(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args)

            weights_path = checkpoint_path.format(net=model_name, shadowset=shadow_index)
            torch.save(net.state_dict(), weights_path)


def train_out_shadow_assmb(args):
    print('training_out_assembly...')
    for model_name in settings.MODEL_LIST:

        for shadow_index in range(settings.PER_MODEL_NUM):

            net = get_network_assmb(model_name, args)
            data, labels = load_data(args, shadow_index)
            #
            # # data preprocessing:
            training_loader = get_dataloader(args, data, labels)
            loss_function = nn.CrossEntropyLoss()
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)
            checkpoint_path = os.path.join(checkpoint_path, 'out')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_path, 'out-{net}-{shadowset}.pth')
            optimizer = optim.Adam(net.parameters())
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
            # # learning rate decay
            iter_per_epoch = len(training_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

            for epoch in range(1, settings.EPOCH[model_name] + 1):
                if epoch > args.warm:
                    train_scheduler.step(epoch)

                train(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args)

            weights_path = checkpoint_path.format(net=model_name, shadowset=shadow_index)
            torch.save(net.state_dict(), weights_path)


def train_test_shadow_assmb(args):
    print('training_target_assembly...')

    for model_name in settings.MODEL_LIST:
        
        for shadow_index in range(settings.PER_MODEL_NUM):

            net = get_network_assmb(model_name, args)
            data, labels = load_target_data(args, shadow_index)

            training_loader = get_dataloader(args, data, labels)
            loss_function = nn.CrossEntropyLoss()
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)
            checkpoint_path = os.path.join(checkpoint_path, 'test')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_path, 'target-{net}-{shadowset}.pth')
            optimizer = optim.Adam(net.parameters())
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
            # # learning rate decay
            iter_per_epoch = len(training_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

            for epoch in range(1, settings.EPOCH[model_name] + 1):
                if epoch > args.warm:
                    train_scheduler.step(epoch)

                train(net, epoch, optimizer, training_loader, loss_function, warmup_scheduler, args)

            weights_path = checkpoint_path.format(net=model_name, shadowset=shadow_index)
            torch.save(net.state_dict(), weights_path)


def shadow_training(args, target_index):
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net).format(dataset=args.dataset, data_size=args.data_size)
    if args.net != 'assembly':
        if not os.path.exists(os.path.join(checkpoint_path, 'out')) or args.over:
            train_out_shadow_model(args)
        if not os.path.exists(os.path.join(checkpoint_path, 'test')) or args.over:
            if args.test_net != 'assembly':
                train_test_shadow_model(args)
                if args.defense == 'adv':
                    tune_adv_target_model(args, args.defense_arg)
                elif args.defense == 'ppb':
                    tune_ppb_target_model(args, args.defense_arg)
            else:
                train_test_shadow_assmb(args)
            if args.is_fine_tuned:
                checkpoint_path = os.path.join(checkpoint_path, 'test', 'target-{net}-{shadowset}.pth'.format(net=args.test_net))
                # fine_tune()
        if args.online:
            train_in_shadow_model(args, target_index)
    else:
        if not os.path.exists(os.path.join(checkpoint_path, 'out')) or args.over:
            train_out_shadow_assmb(args)
        if not os.path.exists(os.path.join(checkpoint_path, 'test')) or args.over:
            train_test_shadow_assmb(args)
        if args.online:
            train_in_shadow_assmb(args, target_index)
    args.over = False


def get_dataloader(args, data, labels):
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        training_dataset = CIFAR100Train(data, labels, transform=transfer)
    elif args.dataset in 'location texas100 purchase100':
        # TODO: finish data process
        training_dataset = TensorDataset(data, labels)
    elif args.dataset in 'gtsrb stl10 svhn':
        training_dataset = OtherTrain(data, labels, transform=transfer)

    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    return training_loader


def gap_atk(args):
    acc1 = 0
    acc2 = 0
    for index in range(args.test_amt):
        target_model = get_network_assmb(args.test_net, args)
        if args.defense == 'adv':
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.test_net, 'test', f'adv_target-{args.test_net}-{index}.pth')
        elif args.defense == 'ppb':
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.test_net, 'test', f'ppb_target-{args.test_net}-{index}.pth')
        else:
            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'assembly', 'test', f'target-{args.test_net}-{index}.pth')
        checkpoint_path = checkpoint_path.format(dataset=args.dataset, data_size=args.test_data_size)
        target_model.load_state_dict(torch.load(checkpoint_path))
        data, labels, test_data, test_labels = load_adv_data(args, index)
        train_loader = get_dataloader(args, data, labels)
        test_loader = get_dataloader(args,test_data, test_labels)
        acc1 += test(target_model, train_loader)
        print(acc1 / (index+1))
        acc2 += test(target_model, test_loader)
        print(acc2 / (index+1))
    return (0.5 + (acc1-acc2)/(2*args.test_amt)) * 100


def test(model, data_loader):
    model.eval()
    train_acc = 0
    for batch_index, (data, labels) in enumerate(data_loader):
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()
        data = data.type(torch.FloatTensor)
        data = data.cuda()
        outputs = model(data)
        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum()
        acc = int(num_correct) / data.shape[0]
        train_acc += acc

    return train_acc/len(data_loader)

