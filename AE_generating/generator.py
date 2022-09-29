import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable


std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]

transfer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def img_2_tensor():
    pass


def get_networks(args):
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net


def get_test_networks(args):
    if args.test_net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.test_net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.test_net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.test_net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.test_net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.test_net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.test_net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.test_net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.test_net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.test_net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.test_net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.test_net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.test_net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.test_net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.test_net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.test_net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.test_net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.test_net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.test_net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.test_net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.test_net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.test_net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.test_net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.test_net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.test_net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.test_net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.test_net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.test_net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.test_net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.test_net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.test_net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.test_net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.test_net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.test_net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.test_net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.test_net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.test_net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.test_net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.test_net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.test_net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.test_net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.test_net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.test_net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.test_net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net


def get_target():
    _loss = None
    _cur_loss = 10000
    _cur_target = 0
    for ltnt_target in range(args.total_label_number):
        _first_flag = True
        for _batch_number in range(int(args.shadow_index / args.model_batch) + 1):
            _in_nets = []
            _out_nets = []
            _batch_lower_bound = batch_number * args.model_batch
            _batch_upper_bound = (batch_lower_bound + args.model_batch) \
                if batch_lower_bound + args.model_batch < args.shadow_index else args.shadow_index
            for _i in range(_batch_lower_bound, _batch_upper_bound):

                _out_net = get_networks(args)
                _out_net.load_state_dict(torch.load(args.state_path + args.net +
                                                    '/out-{target_index}-{net_name}-120-{shadow_index}.pth'
                                                    .format(target_index=target_index,
                                                            net_name=args.net, shadow_index=_i)))

                if device != 'cpu':
                    _out_net = _out_net.cuda()
                _out_net.eval()
                _out_nets.append(_out_net)

                for iter in range(len(_in_nets)):
                    _out_model = _out_nets[iter]
                    _out_output = _out_model(inp)
                    _loss_2 = loss_function(_out_output, ltnt_target)

                    if _first_flag:
                        _loss = args.alpha * _loss_2.item()
                        _first_flag = False
                    else:
                        _loss += args.alpha * _loss_2.item()

        _cur_target = ltnt_target if _cur_loss > _loss else _cur_target
        _cur_loss = _loss if _cur_loss > _loss else _cur_loss

    return _cur_target


def generating_online_aes(ground_class, target_class, in_models, out_models):

    _first_flag = True
    loss = None
    in_pred_list = []
    out_pred_list = []
    for iter in range(len(in_models)):
        in_model = in_models[iter]
        out_model = out_models[iter]
        in_output = in_model(inp)
        out_output = out_model(inp)
        _, in_preds = in_output.max(1)
        _, out_preds = out_output.max(1)
        loss_1 = loss_function(in_output, ground_class)
        loss_2 = loss_function(out_output, target_class)
        mse = mse_loss(inp, ori)
        in_pred_list.append(in_preds.cpu().numpy()[0])
        out_pred_list.append(out_preds.cpu().numpy()[0])

        if _first_flag:
            loss = loss_1 + args.alpha * loss_2
            _first_flag = False
        else:
            loss += loss_1 + args.alpha * loss_2

    loss.backward()
    return inp.grad.data, loss


def generating_offline_aes(target_class, in_models, out_models):

    _first_flag = True
    loss = None
    loss_1 = None
    in_pred_list = []
    out_pred_list = []
    for iter in range(len(out_models)):
        out_model = out_models[iter]
        in_model = in_models[iter]
        out_output = out_model(inp)
        in_output = in_model(inp)
        _, in_preds = in_output.max(1)
        _, out_preds = out_output.max(1)
        loss_1 = mse_loss(inp, ori)
        loss_2 = loss_function(out_output, target_class)
        in_pred_list.append(in_preds.cpu().numpy()[0])
        out_pred_list.append(out_preds.cpu().numpy()[0])

        if _first_flag:
            loss = loss_1 + args.alpha * loss_2
            _first_flag = False
        else:
            loss += loss_1 + args.alpha * loss_2

    loss.backward()
    print('in_label:{}\t, out_label:{}\t MSE_loss:{}'.format(in_pred_list, out_pred_list, loss_1.item()))
    return inp.grad.data, loss


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
    parser.add_argument('-train_amt', type=int, default=20, help='amount of generating models')
    parser.add_argument('-state_path', type=str, default='../shadow_training/checkpoint/', help='path of state dict')
    parser.add_argument('-alpha', type=float, default=1, help='lambda balances loss')
    parser.add_argument('-net', type=str, default='resnet18', help='shadow model type')
    parser.add_argument('-test_net', type=str, default='resnet18', help='test shadow model type')
    parser.add_argument('-if_targeted', type=int, default=True, help='if is targeted')
    parser.add_argument('-target_class', type=int, default=9, help='target label, available when if_targeted is True')
    parser.add_argument('-target_index', type=int, default=30, help='target index number')
    parser.add_argument('-AE_path', type=str, default='./AE_{}/')
    parser.add_argument('-total_label_number', type=int, default=10, help='total numbers of labels')

    parser.add_argument('-model_batch', type=int, default=5, help='batch size for models loaded in one time')

    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-eps', type=int, default=8, help='epsilon')
    parser.add_argument('-gpu', action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_TP = 0
    test_TN = 0
    test_FP = 0
    test_FN = 0
    ground_label = 0
    targeted_label = 0

    for target_index in range(args.target_index):
        target_data = np.load(args.target_path_d)[target_index]
        target_label = np.load(args.target_path_l)[target_index]

        r = target_data[:1024].reshape(32, 32)
        g = target_data[1024:2048].reshape(32, 32)
        b = target_data[2048:].reshape(32, 32)
        inp = np.dstack((r, g, b))
        inp = inp.astype(np.uint8)
        inp = transfer(inp)
        inp = Variable(inp.type(torch.FloatTensor).to(device).unsqueeze(0), requires_grad=True)
        ori = inp.clone()

        target = get_target()
        target_clas = np.array([target])
        target_clas = torch.from_numpy(target_clas)
        target_clas = target_clas.type(torch.LongTensor)
        target_clas = target_clas.to(device)

        target_label = np.array([target_label])
        target_label = torch.from_numpy(target_label)
        target_label = target_label.type(torch.LongTensor)
        target_label = target_label.to(device)

        loss_function = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        AE_grad = None
        AE_loss = None

        for epoch in range(args.epoch):

            first_flag = True

            for batch_number in range(int(args.shadow_index / args.model_batch)+1):

                in_nets = []
                out_nets = []

                batch_lower_bound = batch_number * args.model_batch
                batch_upper_bound = (batch_lower_bound + args.model_batch) if batch_lower_bound + args.model_batch < args.shadow_index else args.shadow_index

                for i in range(batch_lower_bound, batch_upper_bound):

                    out_net = get_networks(args)
                    out_net.load_state_dict(torch.load(args.state_path + args.net +
                                                       '/out-{target_index}-{net_name}-120-{shadow_index}.pth'
                                                       .format(target_index=target_index,
                                                               net_name=args.net, shadow_index=i)))
                    # print('reading' + args.state_path + args.net +
                    #                                    '/out-0-{}-120-{}.pth'.format(args.net, i))

                    if device != 'cpu':
                        out_net = out_net.cuda()
                    out_net.eval()

                    in_net = get_networks(args)
                    in_net.load_state_dict(torch.load(args.state_path + args.net +
                                                      '/in-{target_index}-{net_name}-120-{shadow_index}.pth'
                                                      .format(target_index=target_index,
                                                              net_name=args.net, shadow_index=i)))
                    # print('reading' + args.state_path + args.net +
                    #       '/out-0-{}-120-{}.pth'.format(args.net, i))

                    if device != 'cpu':
                        in_net = in_net.cuda()
                    in_net.eval()

                    in_nets.append(in_net)
                    out_nets.append(out_net)


                grad, loss = generating_online_aes(ground_class=target_label , target_class=target_clas, in_models=in_nets, out_models=out_nets)
                # grad, loss = generating_offline_aes(target_class=target_clas, in_models=in_nets, out_models=out_nets)

                if not first_flag:
                    AE_grad += grad
                    AE_loss += loss
                else:
                    AE_grad, AE_loss = grad.clone(), loss.clone()
                    first_flag = False

                inp.grad.data.zero_()

            inp.data -= args.lr * AE_grad

            print('Generate epoch:{epoch}\tloss:{}'.format(AE_loss.item(), epoch=epoch))

        for i in range(args.train_amt, args.shadow_index):

            out_net_t = get_test_networks(args)
            out_net_t.load_state_dict(torch.load(args.state_path + args.net +
                                                 '/out-{target_index}-{net_name}-120-{shadow_index}.pth'
                                                 .format(target_index=target_index,
                                                         net_name=args.net, shadow_index=i)))

            if args.device != 'cpu':
                out_net_t = out_net_t.cuda()
            out_net_t.eval()

            in_net_t = get_test_networks(args)
            in_net_t.load_state_dict(torch.load(args.state_path + args.net +
                                                '/in-{target_index}-{net_name}-120-{shadow_index}.pth'
                                                .format(target_index=target_index,
                                                        net_name=args.net, shadow_index=i)))
            if args.device != 'cpu':
                in_net_t = in_net_t.cuda()
            in_net_t.eval()

            out_output = out_net_t(inp)
            in_output = in_net_t(inp)

            _, in_preds = in_output.max(1)
            _, out_preds = out_output.max(1)

            in_pd = in_preds.cpu().numpy()[0]
            out_pd = out_preds.cpu().numpy()[0]

            if out_pd == ground_label:
                test_FP += 1
            else:
                test_TN += 1
            if in_pd == ground_label:
                test_TP += 1
            else:
                test_FP += 1

        test_acc = (test_TP + test_TN) / (test_TP + test_FP + test_TN + test_FN) * 100
        test_pre = test_TP / (test_FP + test_TP) * 100
        test_rcl = test_TP / (test_TP + test_FN) * 100

        print('test: ACC:{test_acc}, Precision:{test_pre}, test_rcl:{test_rcl}'
              .format(test_acc=test_acc, test_pre=test_pre, test_rcl=test_rcl))

