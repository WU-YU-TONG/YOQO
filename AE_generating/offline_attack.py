import sys

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable

from generator import get_networks, get_test_networks

std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]

transfer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def generating_offline_aes(target_class, out_models):
    out_output = out_models(inp)
    _, out_preds = out_output.max(1)
    loss_1 = mse_loss(inp, ori)
    loss_2 = loss_function(out_output, target_class)

    loss = loss_1 + args.alpha * loss_2

    loss.backward()
    return inp.grad.data, loss


def get_multi_target():
    _cur_target = []

    for _batch_number in tqdm(range(args.train_amt)):
        _out_net = get_networks(args)
        _out_net.load_state_dict(torch.load(args.state_path +
                                            '/out-0-{net_name}-120-{shadow_index}.pth'
                                            .format(net_name=args.net, shadow_index=_batch_number)))

        _out_net = _out_net.cuda()
        _out_net.eval()
        _out_model = _out_net
        _out_output = _out_model(inp)
        _out_output = _out_output.detach().cpu().numpy()
        _temp_target_list = np.argsort(_out_output)[0]
        _temp_target = _temp_target_list[-1] if _temp_target_list[-1] != target_label else _temp_target_list[-2]
        _temp_target = np.array([_temp_target])
        _temp_target = torch.from_numpy(_temp_target)
        _temp_target = _temp_target.type(torch.LongTensor)
        _temp_target = _temp_target.to(device)
        _cur_target.append(_temp_target.clone())

    return _cur_target


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('-epoch', type=int, default=15, help='training epoch')
    parser.add_argument('-target_path', type=str, default='images/target.jpg', help='path to image')
    parser.add_argument('-target_path_d', type=str, default='../shadow_dataset/test_data.npy',
                        help='path of no-allocated data')
    parser.add_argument('-target_path_l', type=str, default='../shadow_dataset/test_labels.npy',
                        help='path of no-allocated labels')
    parser.add_argument('-shadow_index', type=int, default=25, help='index of shadow dataset')
    parser.add_argument('-train_amt', type=int, default=19, help='amount of generating models')
    parser.add_argument('-test_amt', type=int, default=5, help='amount of tested models')
    parser.add_argument('-state_path', type=str, default='./test_model/', help='path of state dict')
    parser.add_argument('-alpha', type=float, default=1, help='lambda balances loss')
    parser.add_argument('-net', type=str, default='resnet18', help='shadow model type')
    parser.add_argument('-test_net', type=str, default='resnet18', help='test shadow model type')
    parser.add_argument('-test_net_path', type=str, default='./test_model/test-0-{net}-{test_training_epoch}-{index}.pth', help='path to test model')
    parser.add_argument('-if_targeted', type=int, default=True, help='if is targeted')
    parser.add_argument('-target_index', type=int, default=1000, help='target index number')
    parser.add_argument('-AE_path', type=str, default='./AE_Offline/')
    parser.add_argument('-is_multilabel', type=int, default=True, help='if use multi-label technique')
    parser.add_argument('-total_label_number', type=int, default=10, help='total numbers of labels')

    parser.add_argument('-model_batch', type=int, default=2, help='batch size for models loaded in one time')

    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-eps', type=int, default=8, help='epsilon')
    parser.add_argument('-gpu', action="store_true", default=False)
    parser.add_argument('-loss_threshold', type=float, default=10, help='threshold of generating loss')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_TP = 869
    test_TN = 745
    test_FP = 335
    test_FN = 211
    ground_label = 0
    targeted_label = 0
    loss_function = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    training_epoch = 120 if args.net == 'resnet18' else 150
    test_training_epoch = 120 if args.net == 'resnet18' else 150

    for target_index in range(432, args.target_index):
        target_data = np.load(args.target_path_d)[target_index]
        target_label = np.load(args.target_path_l)[target_index]
        ground_label = np.load(args.target_path_l)[target_index]

        r = target_data[:1024].reshape(32, 32)
        g = target_data[1024:2048].reshape(32, 32)
        b = target_data[2048:].reshape(32, 32)
        inp = np.dstack((r, g, b))
        inp = inp.astype(np.uint8)
        inp = transfer(inp)
        inp = Variable(inp.type(torch.FloatTensor).to(device).unsqueeze(0), requires_grad=True)
        ori = inp.clone()

        target_clas = get_multi_target()

        target_label = np.array([target_label])
        target_label = torch.from_numpy(target_label)
        target_label = target_label.type(torch.LongTensor)
        target_label = target_label.to(device)

        AE_grad = None
        AE_loss = None

        fi_loss = args.loss_threshold + 1

        # while fi_loss > args.loss_threshold:
        for _ in range(args.epoch):

            first_flag = True

            for i in range(args.train_amt):

                out_net = get_networks(args)
                out_net.load_state_dict(torch.load(args.state_path +
                                                   '/out-0-{net_name}-{training_epoch}-{shadow_index}.pth'
                                                   .format(net_name=args.net, training_epoch=training_epoch,
                                                           shadow_index=i)))

                if device != 'cpu':
                    out_net = out_net.cuda()
                out_net.eval()

                if args.is_multilabel:
                    grad, loss = generating_offline_aes(target_class=target_clas[i], out_models=out_net)
                else:
                    grad, loss = generating_offline_aes(target_class=target_clas, out_models=out_net)

                if not first_flag:
                    AE_grad += grad
                    AE_loss += loss
                else:
                    AE_grad, AE_loss = grad.clone(), loss.clone()
                    first_flag = False

                inp.grad.data.zero_()

            inp.data -= args.lr * AE_grad

            print('Generate epoch:\tloss:{}'.format(AE_loss.item()))

            fi_loss = AE_loss.item()

        for test_index in range(args.test_amt):

            net_t = get_test_networks(args)
            net_t.load_state_dict(torch.load(args.test_net_path.format(net=args.test_net,
                                                                       test_training_epoch=test_training_epoch,
                                                                       index=test_index)))
            if args.device != 'cpu':
                net_t = net_t.cuda()
            net_t.eval()

            t_output = net_t(inp)
            _, t_preds = t_output.max(1)
            t_pd = t_preds.cpu().numpy()[0]

            if target_index % 2 == 0:
                if t_pd == ground_label:
                    test_TP += 1
                else:
                    test_FN += 1
            else:
                if t_pd == ground_label:
                    test_FP += 1
                else:
                    test_TN += 1

        np.save(args.AE_path + '{}.npy'.format(target_index), inp.cpu().detach().numpy())

        test_acc = (test_TP + test_TN) / (test_TP + test_FP + test_TN + test_FN) * 100
        test_pre = test_TP / (test_FP + test_TP) * 100
        test_rcl = test_TP / (test_TP + test_FN) * 100
        print('FP:{}\tTN:{}\tTP:{}\tFN:{}'.format(test_FP, test_TN, test_TP, test_FN))
        print('test: ACC:{test_acc}, Precision:{test_pre}, test_rcl:{test_rcl}'
              .format(test_acc=test_acc, test_pre=test_pre, test_rcl=test_rcl))



