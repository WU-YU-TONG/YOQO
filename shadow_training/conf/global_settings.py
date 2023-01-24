import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = './checkpoint/{dataset}/{data_size}/'
ONLINE_AE_PATH = './AE_Online/{dataset}/{data_size}/{net}/'
OFFLINE_AE_PATH = './AE_Offline/{dataset}/{data_size}/{net}/'

SHADOW_DATA_PATH = '../shadow_dataset/{dataset}/{data_size}/shadow_data_{shadow_index}.npy'
SHADOW_LABELS_PATH = '../shadow_dataset/{dataset}/{data_size}/shadow_labels_{shadow_index}.npy'
TARGET_DATA_PATH = '../shadow_dataset/{dataset}/{data_size}/target_data.npy'
TARGET_LABELS_PATH = '../shadow_dataset/{dataset}/{data_size}/target_labels.npy'
TEST_DATA_PATH = '../shadow_dataset/{dataset}/test_data.npy'
TEST_LABELS_PATH = '../shadow_dataset/{dataset}/test_labels.npy'

#dataset config
DATASET_CFG = {
    'CIFAR10':{'model': 'image_model', 'num_cls': 10, 'input_dim': (32, 32, 3)},
    'CIFAR100':{'model': 'image_model', 'num_cls': 100, 'input_dim': (32, 32, 3)},
    'gtsrb':{'model': 'image_model', 'num_cls': 43, 'input_dim': (32, 32, 3)},
    'svhn':{'model': 'image_model', 'num_cls': 10, 'input_dim': (32, 32, 3)},
    'texas100': {'model': 'ColumnFC', 'num_cls': 100, 'input_dim': 6169},
    'location': {'model': 'ColumnFC', 'num_cls': 30, 'input_dim': 446},
    'purchase100': {'model': 'ColumnFC', 'num_cls': 100, 'input_dim': 600}
}



#total training epoches
EPOCH = {'ColumnFC': 30,
         'CNN7': 30,
         'resnet18': 20,
         'densenet121': 30,
         'vgg16': 50,
         'inceptionv3': 25,
         'seresnet18': 20}
ADV_TOTAL_EPOCH = 50
ADV_TUNING_LR = 0.001
MILESTONES = [20, 40]


#settings for experiment
MODEL_LIST = ['vgg16', 'resnet18', 'CNN7', 'densenet121', 'inceptionv3', 'seresnet18']
TEST_MODEL_LIST = ['vgg16', 'resnet18', 'CNN7', 'densenet121', 'inceptionv3', 'seresnet18']
PER_MODEL_NUM = 4
ALPHA_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
TRAIN_AMT_LIST = [1, 2, 4, 8, 16, 32]










