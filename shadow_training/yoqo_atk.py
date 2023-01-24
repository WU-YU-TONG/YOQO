import sys
import os

import numpy as np
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable

from atk_utils import shadow_training
from utils import get_network_assmb, test_sample_process
from conf import settings
from defenses import memguard


def get_multi_target(train_amt):
    _cur_target = []
   
    for _name in TRAIN_MODEL_NAMES:
        for _index in range(train_amt):
            _out_net = get_network_assmb(_name, args)
            _out_net.load_state_dict(torch.load((settings.CHECKPOINT_PATH + args.net +
                                                '/out/out-{net_name}-{shadow_index}.pth')
                                                .format(net_name=_name, dataset=args.dataset, shadow_index=_index, data_size=args.data_size)))

            if args.device != 'cpu':
                _out_net = _out_net.cuda()
            _out_net.eval()
            _out_model = _out_net
            _out_output = _out_model(inp)
            _out_output = _out_output.detach().cpu().numpy()
            _temp_target_list = np.argsort(_out_output)[0]
            _temp_target = _temp_target_list[-1] if _temp_target_list[-1] != ground_label else _temp_target_list[-2]
            _temp_target = np.array([_temp_target])
            _temp_target = torch.from_numpy(_temp_target)
            _temp_target = _temp_target.type(torch.LongTensor)
            _temp_target = _temp_target.cuda()
            _cur_target.append(_temp_target.clone())    

    return _cur_target


def generating_online_aes(ground_class, target_class, in_model, out_model, loss_func, alpha):
    in_output = in_model(inp)
    out_output = out_model(inp)
    _, in_preds = in_output.max(1)
    _, out_preds = out_output.max(1)
    loss_1 = loss_func(in_output, ground_class)
    loss_2 = loss_func(out_output, target_class)

    loss = (loss_1 + alpha * loss_2) / (1+alpha) * 3
    
    loss.backward()
    return inp.grad.data, loss


def generating_offline_aes(ground_class, target_class, out_model, in_model, loss_func, alpha):
    out_output = out_model(inp)
    _, out_preds = out_output.max(1)
    loss_1 = mse_loss(inp, ori)
    loss_2 = loss_function(out_output, target_class)

    # loss = loss_1 if out_preds == target_class else loss_2
    loss = (alpha * loss_1 + loss_2) / (1+alpha) * 6
    loss.backward()
    # print(loss_2.item())
    return inp.grad.data, loss


def attack(args, train_model_names, atk_mode, loss_func, target_label, alpha, train_amt):
    # inp = ori.clone()
    AE_grad = None
    AE_loss = None
    epoch = 0
    fi_loss = args.loss_threshold + 1

    while(fi_loss > args.loss_threshold):

        first_flag = True

        for model_name in train_model_names:

            for i in range(train_amt):
                
                out_model_checkpoint = os.path.join(settings.CHECKPOINT_PATH, args.net, 'out', 'out-{net_name}-{shadow_index}.pth')
                out_model_checkpoint = out_model_checkpoint.format(net_name=model_name, dataset=args.dataset, shadow_index=i, data_size=args.data_size)
                out_net = get_network_assmb(model_name, args)
                out_net.load_state_dict(torch.load(out_model_checkpoint))

                if args.device != 'cpu':
                    out_net = out_net.cuda()
                out_net.eval()
                
                in_net = None
                if args.online:
                    in_model_checkpoint = os.path.join(settings.CHECKPOINT_PATH, args.net, 'in', 'in-{net_name}-{shadow_index}.pth')
                    in_model_checkpoint = in_model_checkpoint.format(net_name=model_name, dataset=args.dataset, shadow_index=i, data_size=args.data_size)
                    in_net = get_network_assmb(model_name, args)
                    in_net.load_state_dict(torch.load(in_model_checkpoint))

                    if args.device != 'cpu':
                        in_net = in_net.cuda()
                    in_net.eval()

                if args.is_multilabel:
                    grad, loss = atk_mode(ground_class=target_label, target_class=target_clas[i],
                                            out_model=out_net, in_model=in_net, loss_func=loss_func, alpha=alpha)
                else:
                    grad, loss = atk_mode(ground_class=target_label, target_class=target_clas, 
                                            out_model=out_net, in_model=in_net, loss_func=loss_func, alpha=alpha)

                if not first_flag:
                    AE_grad += grad
                    AE_loss += loss
                else:
                    AE_grad, AE_loss = grad.clone(), loss.clone()
                    first_flag = False

                inp.grad.data.zero_()

        inp.data -= args.lr * AE_grad
        #normalizing the loss
        fi_loss = AE_loss.item() / train_amt * 16
        print('Generate epoch:\tloss:{}'.format(fi_loss))
        epoch += 1
        if epoch > args.max_epoch:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    parser.add_argument('-dataset', type=str, default='CIFAR10', help='net type')
    parser.add_argument('-train_amt', type=int, default=16, help='amount of generating models')
    parser.add_argument('-test_amt', type=int, default=5, help='amount of tested models')
    parser.add_argument('-alpha', type=float, default=5, help='lambda balances loss')
    parser.add_argument('-net', type=str, default='CNN7', help='shadow model type, fixed to ColumnFC for location, texas100 and Purchase100')
    parser.add_argument('-test_net', type=str, default='CNN7', help='test shadow model type')
    parser.add_argument('-target_index', type=int, default=400, help='target index number, should be less than 1000')
    parser.add_argument('-test_net_path', type=str, default='../shadow_training/checkpoint/{data_size}/{test_net}/test/target-{net}-{index}.pth', help='path to test model')
    parser.add_argument('-is_multilabel', type=bool, default=True, help='if use multi-label technique')

    parser.add_argument('-max_epoch', type=int, default=50, help='epoch limitation for attack')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate for attack')
    parser.add_argument('-eps', type=int, default=8, help='epsilon')
    parser.add_argument('-loss_threshold', type=float, default=6, 
                        help='threshold of generating loss')
    parser.add_argument('-online', type=int, default=False, help='Attack mode')
    parser.add_argument('-gpu', action="store_true", default=True)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    
    
    parser.add_argument('-defense', type=str, default='', choices=['adv', 'ppb', 'memguard'])

    parser.add_argument('-defense_arg', type=float, default=1)

    parser.add_argument('-batch_size', type=int, default=128, 
                        help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, 
                       help='warm up training phase')
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-is_load', type=int, default=False, help='if use trained model')
    parser.add_argument('-is_add', type=int, default=True, help='if the model is trained on dataset added candidates')

    parser.add_argument('-data_size', type=int, default=2500, help='size of training set')
    parser.add_argument('-test_data_size', type=int, default=2500, help='size of shadow training set')

    parser.add_argument('-dif_data_size', action='store_true', help='if to use different dataset sizes')
    parser.add_argument('-alpha_scanning', action='store_true')
    parser.add_argument('-assembly_size_scanning', action='store_true', help='parameter scanning on assembly size, CANNOT operate when net=assembly')

    parser.add_argument('-is_fine_tuned', type=int, default=False, help='if fine-tune rather than shadow training')
    parser.add_argument('-over', action='store_true')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if settings.DATASET_CFG[args.dataset]['model'] != 'image_model':
        args.net = settings.DATASET_CFG[args.dataset]['model']
        args.test_net = settings.DATASET_CFG[args.dataset]['model']
        
    checkpoint_path = (settings.CHECKPOINT_PATH+args.net).format(dataset=args.dataset, data_size=args.data_size)
    if not args.dif_data_size:
        args.test_data_size = args.data_size
    
    TRAIN_MODEL_NAMES = settings.MODEL_LIST if args.net == 'assembly' else [args.net]
    TRAIN_AMT = settings.PER_MODEL_NUM if args.net == 'assembly' else args.train_amt
    TEST_MODEL_NAMES = settings.TEST_MODEL_LIST if args.test_net == 'assembly' else [args.test_net]    
    TEST_AMT = settings.PER_MODEL_NUM if args.test_net == 'assembly' else args.test_amt

    if args.alpha_scanning:
        RECORD_KEYS = settings.ALPHA_LIST
        PARAM_SWEEP_INTERVAL = settings.ALPHA_LIST
        SWEEP_MODE = 'alpha_sweep'
    elif args.assembly_size_scanning:
        RECORD_KEYS = settings.TRAIN_AMT_LIST
        PARAM_SWEEP_INTERVAL = settings.TRAIN_AMT_LIST
        SWEEP_MODE = 'assembly_sweep'
    else:
        RECORD_KEYS = TEST_MODEL_NAMES 
        PARAM_SWEEP_INTERVAL = [1]
        SWEEP_MODE = 'None'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        f = open((checkpoint_path+'/{net}-{testnet}-{mode}-{sweep}-record.txt')
                  .format(net=args.net, testnet=args.test_net, mode=args.online, sweep=SWEEP_MODE), 'w')
        f.writelines(f'======================{args.test_net}=={args.net}=={args.loss_threshold}=={args.alpha}=={args.test_data_size}===================\n')
    else:
        with open((checkpoint_path+'/{net}-{testnet}-{mode}-{sweep}-record.txt')
                    .format(net=args.net, testnet=args.test_net, mode=args.online, sweep=SWEEP_MODE), 'a') as f:
            f.writelines(f'========================{args.test_net}=={args.net}=={args.loss_threshold}=={args.alpha}=={args.test_data_size}===================\n')
    f.close()

    if not args.resume:
        test_TP = {model_name: 0 for model_name in RECORD_KEYS}
        test_TN = {model_name: 0 for model_name in RECORD_KEYS}
        test_FP = {model_name: 0 for model_name in RECORD_KEYS}
        test_FN = {model_name: 0 for model_name in RECORD_KEYS}
        start_point = 0
    else:
        resume_point = np.load(os.path.join(checkpoint_path, '{test_model}-{mode}-{sweep}-resume_point.npy'
                                .format(test_model=args.test_net, mode=args.online, sweep=SWEEP_MODE)), allow_pickle=True).item()
        test_TP = resume_point['TP']
        test_TN = resume_point['TN']
        test_FP = resume_point['FP']
        test_FN = resume_point['FN']
        start_point = resume_point['current_index'] + 1

    ground_label = 0
    targeted_label = 0
    loss_function = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    for target_index in range(start_point, args.target_index):
        shadow_training(args, target_index)
        
        for cur_param in PARAM_SWEEP_INTERVAL:
            
            inp, ori, ground_label, target_label = test_sample_process(args, target_index)
            
            if args.assembly_size_scanning:
                target_clas = get_multi_target(cur_param)
            else:
                target_clas = get_multi_target(TRAIN_AMT)

            if args.online:
                attack(args, TRAIN_MODEL_NAMES, generating_online_aes, loss_func=loss_function, 
                            target_label=target_label,
                            alpha=cur_param if args.alpha_scanning else args.alpha,
                            train_amt=cur_param if args.assembly_size_scanning else TRAIN_AMT)
            else:
                attack(args, TRAIN_MODEL_NAMES, generating_offline_aes, loss_func=loss_function, 
                            target_label=target_label,
                            alpha=cur_param if args.alpha_scanning else args.alpha,
                            train_amt=cur_param if args.assembly_size_scanning else TRAIN_AMT)

            for model_name in TEST_MODEL_NAMES:
                if args.defense == 'adv':
                    test_model_checkpoint = os.path.join(checkpoint_path, 'test', 'adv_target-{net_name}-{shadow_index}.pth')
                elif args.defense == 'pbb':
                    test_model_checkpoint = os.path.join(checkpoint_path, 'test', 'ppb_target-{net_name}-{shadow_index}.pth')
                else:
                    test_model_checkpoint = os.path.join(checkpoint_path, 'test', 'target-{net_name}-{shadow_index}.pth')

                for test_index in range(TEST_AMT):

                    cur_test_model_checkpoint = test_model_checkpoint.format(net_name=model_name, shadow_index=test_index, data_size=args.data_size)
                    net_t = get_network_assmb(model_name, args)

                    net_t.load_state_dict(torch.load(cur_test_model_checkpoint))
                    if args.device != 'cpu':
                        net_t = net_t.cuda()
                    net_t.eval()

                    t_output = net_t(inp)

                    if args.defense == 'memguard':
                        t_output = memguard(t_output)

                    _, t_preds = t_output.max(1)
                    t_pd = t_preds.cpu().numpy()[0]

                    if args.assembly_size_scanning or args.alpha_scanning:
                        record_key = cur_param
                    else:
                        record_key = model_name

                    if target_index % 2 == 0:
                        if t_pd == ground_label:
                            test_TP[record_key] += 1
                        else:
                            test_FN[record_key] += 1
                    else:
                        if t_pd == ground_label:
                            test_FP[record_key] += 1
                        else:
                            test_TN[record_key] += 1
            
                test_acc = (test_TP[record_key] + test_TN[record_key]) / (test_TP[record_key] +
                                        test_FP[record_key] + test_TN[record_key] + test_FN[record_key]) * 100
                test_pre = test_TP[record_key] / (test_FP[record_key] + test_TP[record_key] + 0.00001) * 100
                test_rcl = test_TP[record_key] / (test_TP[record_key] + test_FN[record_key] + 0.00001) * 100
                print(record_key)
                print('FP:{}\tTN:{}\tTP:{}\tFN:{}'.format(test_FP[record_key], test_TN[record_key], test_TP[record_key], test_FN[record_key]))
                print('test: ACC:{test_acc}, Precision:{test_pre}, test_rcl:{test_rcl}'
                    .format(test_acc=test_acc, test_pre=test_pre, test_rcl=test_rcl))
                with open((checkpoint_path+'/{net}-{testnet}-{mode}-{sweep}-record.txt')
                    .format(net=args.net, testnet=args.test_net, data_size=args.data_size, mode=args.online, sweep=SWEEP_MODE), 'a') as f:
                    f.writelines(str(record_key) + ':\n')
                    f.writelines('FP:{}\tTN:{}\tTP:{}\tFN:{}\n'.format(test_FP[record_key], test_TN[record_key], test_TP[record_key], test_FN[record_key]))
                    f.writelines('test: ACC:{test_acc}, Precision:{test_pre}, Test_rcl:{test_rcl}\n'
                                .format(test_acc=test_acc, test_pre=test_pre, test_rcl=test_rcl))
                    f.close()

        if not (args.alpha_scanning or args.assembly_size_scanning):
            if args.online:
                AE_path = (settings.ONLINE_AE_PATH + '{target_index}.npy').format(target_index=target_index, dataset=args.dataset, data_size=args.data_size, net=args.net)
                if not os.path.exists(settings.ONLINE_AE_PATH.format(data_size=args.data_size, dataset=args.dataset, net=args.net)):
                    os.makedirs(settings.ONLINE_AE_PATH.format(data_size=args.data_size, dataset=args.dataset, net=args.net))
            else:
                AE_path = (settings.OFFLINE_AE_PATH + '{target_index}.npy').format(target_index=target_index, dataset=args.dataset, data_size=args.data_size, net=args.net)
                if not os.path.exists(settings.OFFLINE_AE_PATH.format(data_size=args.data_size, dataset=args.dataset, net=args.net)):
                    os.makedirs(settings.OFFLINE_AE_PATH.format(data_size=args.data_size, dataset=args.dataset, net=args.net))
            np.save(AE_path, inp.cpu().detach().numpy())

        np.save(os.path.join(checkpoint_path, '{test_model}-{mode}-{sweep}-resume_point.npy'
                                            .format(test_model=args.test_net, mode=args.online, sweep=SWEEP_MODE)),
                                            {'current_index': target_index,
                                                        'TN': test_TN,
                                                        'TP': test_TP,
                                                        'FN': test_FN,
                                                        'FP': test_FP})
        


