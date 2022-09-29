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

transfer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
image_path = './img_adv_3200.jpg'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('-target_path', type=str, default='images/target.jpg', help='path to image')
parser.add_argument('-target_path_d', type=str, default='../shadow_dataset/targets_data.npy',
                    help='path of no-allocated data')
parser.add_argument('-target_path_l', type=str, default='../shadow_dataset/targets_labels.npy',
                    help='path of no-allocated labels')
parser.add_argument('-state_path', type=str, default='../shadow_training/checkpoint/', help='path of state dict')
parser.add_argument('-alpha', type=float, default=1, help='lambda balances loss')
parser.add_argument('-net', type=str, default='resnet18', help='shadow model type')
parser.add_argument('-if_targeted', type=int, default=True, help='if is targeted')
parser.add_argument('-target_class', type=int, default=1, help='target label, available when if_targeted is True')
parser.add_argument('-ground_class', type=int, default=1, help='ground-truth label of the target')
parser.add_argument('-lr', type=float, default=10, help='learning rate')
parser.add_argument('-eps', type=int, default=8, help='epsilon')
parser.add_argument('-gpu', action="store_true", default=False)
args = parser.parse_args()

out_net = get_networks(args)
out_net.load_state_dict(torch.load(args.state_path + args.net +
                                   '/Monday_12_September_2022_10h_35m_20s/resnet18-140-regular.pth'))

if args.device != 'cpu':
    out_net = out_net.cuda()
out_net.eval()

in_net = get_networks(args)
in_net.load_state_dict(torch.load(args.state_path + args.net +
                                  '/Monday_12_September_2022_11h_05m_25s/resnet18-110-regular.pth'))

if args.device != 'cpu':
    in_net = in_net.cuda()
in_net.eval()


inp = cv2.imread(image_path)[..., ::-1]
inp = Variable(inp.type(torch.FloatTensor).to(device).unsqueeze(0), requires_grad=True)
out_output = out_net(inp)
in_output = in_net(inp)
_, in_preds = in_output.max(1)
_, out_preds = out_output.max(1)

print('in-output:{}, out-output:{}'.format(in_preds, out_preds))
