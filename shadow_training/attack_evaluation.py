import sys
import os

import numpy as np
import torch
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable

from atk_utils import shadow_training, tune_adv_target_model, tune_ppb_target_model, gap_atk, train_test_dp_model
from utils import get_network_assmb
from conf import settings
from defenses import memguard


'''This code uses the consequences of yoqo_atk.py, therefor it cannot operate without 
running yoqo.atk.py in advance.
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('-dataset', type=str, default='CIFAR10', help='dataset used')
    parser.add_argument('-data_size', type=int, default=2500, help='size of dataset used to train the models')
    parser.add_argument('-target_index', type=int, default=400)
    parser.add_argument('-atk_type', type=str, default='online')
    parser.add_argument('-net', type=str, default='CNN7')
    parser.add_argument('-test_net', type=str, default='CNN7')
    parser.add_argument('-test_amt', type=int, default=4)
    parser.add_argument('-defense', type=str, default='DPSGD')
    parser.add_argument('-warm', type=int, default=1, 
                       help='warm up training phase')
    parser.add_argument('-gpu', action="store_true", default=True) 
    parser.add_argument('-test_data_size', type=int, default=2500)
    parser.add_argument('-is_add', type=int, default=True)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-dif_data_size', type=int, default=False)
    parser.add_argument('-defense_arg', type=float, default=0.5)
    parser.add_argument('-over', action='store_true')
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate for attack')
    parser.add_argument('-gap_atk', action='store_true')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if settings.DATASET_CFG[args.dataset]['model'] != 'image_model':
        args.net = settings.DATASET_CFG[args.dataset]['model']
        args.test_net = settings.DATASET_CFG[args.dataset]['model']

    checkpoint_path = (settings.CHECKPOINT_PATH+args.test_net).format(dataset=args.dataset, data_size=args.test_data_size)

    TEST_MODEL_NAMES = settings.TEST_MODEL_LIST if args.test_net == 'assembly' else [args.test_net]    
    TEST_AMT = settings.PER_MODEL_NUM if args.test_net == 'assembly' else args.test_amt

    RECORD_KEYS = TEST_MODEL_NAMES 

    test_TP = {model_name: 0 for model_name in RECORD_KEYS}
    test_TN = {model_name: 0 for model_name in RECORD_KEYS}
    test_FP = {model_name: 0 for model_name in RECORD_KEYS}
    test_FN = {model_name: 0 for model_name in RECORD_KEYS}

    ae_path = settings.ONLINE_AE_PATH if args.atk_type == 'online' else settings.OFFLINE_AE_PATH
    ae_path = ae_path.format(dataset=args.dataset, data_size=args.data_size, net=args.net)
    if args.gap_atk:
        ASR = gap_atk(args)
        print(f'GAP ASR:{ASR}')
        exit(0)
    for index in range(args.target_index):
        path = os.path.join(ae_path, '{}.npy'.format(index))
        if not os.path.exists(path):
            print(f'There is only {index} AEs.')
            break
        inp = torch.tensor(np.load(path)[0])
        inp = Variable(inp.type(torch.FloatTensor).cuda().unsqueeze(0), requires_grad=False)
        ground_label = np.load(settings.TEST_LABELS_PATH.format(dataset=args.dataset))[index]
        for model_name in TEST_MODEL_NAMES:
            if args.defense == 'adv':
                test_model_checkpoint = os.path.join(checkpoint_path, 'test', 'adv_target-{net_name}-{shadow_index}.pth')
            elif args.defense == 'ppb':
                test_model_checkpoint = os.path.join(checkpoint_path, 'test', 'ppb_target-{net_name}-{shadow_index}.pth')
            elif args.defense == 'DPSGD':
                test_model_checkpoint = os.path.join(checkpoint_path, 'test', 'dp-target-{net_name}-{shadow_index}-{arg}.pth')
            else:
                test_model_checkpoint = os.path.join(checkpoint_path, 'test', 'target-{net_name}-{shadow_index}.pth')

            for test_index in range(TEST_AMT):

                cur_test_model_checkpoint = test_model_checkpoint.format(net_name=model_name, shadow_index=test_index, data_size=args.test_data_size, arg=args.defense_arg)
                if not os.path.exists(cur_test_model_checkpoint) or args.over:
                    if args.defense == 'adv':
                        tune_adv_target_model(args, args.defense_arg)
                    elif args.defense == 'ppb':
                        tune_ppb_target_model(args, args.defense_arg)
                    elif args.defense == 'DPSGD':
                        train_test_dp_model(args)
                    args.over = False

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

                record_key = model_name

                if index % 2 == 0:
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