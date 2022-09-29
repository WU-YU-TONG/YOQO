
''' TO RUN THIS FILE, FIRSTLY EXTRACT THE FILE 'CIFAR-10-PYTHON.TAR.GZ'
'''

import numpy as np
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def CreatData():
    x = []
    y = []
    for i in range(1, 6):
        batch_patch = 'cifar-10-batches-py\data_batch_%d' % (i)
        batch_dict = unpickle(batch_patch)
        train_batch=batch_dict[b'data'].astype('float')
        train_labels = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_labels)

    train_data = np.concatenate(x)
    train_labels = np.concatenate(y)

    testpath = os.path.join('cifar-10-batches-py', 'test_batch')
    test_dict = unpickle(testpath)
    test_data = test_dict[b'data'].astype('float')
    test_labels = np.array(test_dict[b'labels'])

    return train_data, train_labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = CreatData()
np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_labels)
np.save('../shadow_dataset/test_data.npy', test_data)
np.save('../shadow_dataset/test_labels.npy', test_labels)
