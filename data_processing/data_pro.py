import numpy as np
import random as rd
import os
DATA_SIZE = 2500
DATA_SET = 'CIFAR10'
SHADOW_AMT = 16
DATA_DIR = f'../shadow_dataset/{DATA_SET}/'
SAVE_DIR = f'../shadow_dataset/{DATA_SET}/{DATA_SIZE}/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if DATA_SET == 'CIFAR10' or DATA_SET == 'CIFAR100':
    a = np.load(os.path.join(DATA_DIR, 'train_data.npy'))
    b = np.load(os.path.join(DATA_DIR, 'train_labels.npy'))
    print(a.shape)
    for i in range(SHADOW_AMT):
        lists = rd.sample(range(50000), DATA_SIZE)
        # lists = list(range(0,3000,2))
        c = a[lists]
        d = b[lists]
        np.save(SAVE_DIR + '/shadow_data_{i}.npy'.format(i=i), c)
        np.save(SAVE_DIR + '/shadow_labels_{i}.npy'.format(i=i), d)

    a = np.load(os.path.join(DATA_DIR, 'test_data.npy'))
    b = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))

    for i in range(5):
        lists = list(range(0,1000,2))
        c = a[lists]
        d = b[lists]
        lists = rd.sample(range(1000, 10000), DATA_SIZE - 500)
        ap = a[lists]
        bp = b[lists]
        c = np.concatenate((c, ap), axis=0)
        d = np.concatenate((d, bp), axis=0)
        np.save(SAVE_DIR + '/target_data_{}.npy'.format(i), c)
        np.save(SAVE_DIR + '/target_labels_{}.npy'.format(i), d)

elif DATA_SET == 'gtsrb':
    a = np.load(os.path.join(DATA_DIR, 'test_data.npy'))
    b = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    print(a.shape)
    for i in range(SHADOW_AMT):
        lists = rd.sample(range(DATA_SIZE + 500, a.shape[0]), DATA_SIZE)
        c = a[lists]
        d = b[lists].reshape(-1)
        np.save(SAVE_DIR + '/shadow_data_{i}.npy'.format(i=i), c)
        np.save(SAVE_DIR + '/shadow_labels_{i}.npy'.format(i=i), d)

    a = np.load(os.path.join(DATA_DIR, 'test_data.npy'))
    b = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    print(b.shape)
    b = b.reshape(-1)
    np.save(os.path.join(DATA_DIR, 'test_labels.npy'), b)
    for i in range(1):
        lists = list(range(0,1000,2))
        lists.extend(list(range(1000, DATA_SIZE + 500)))
        c = a[lists]
        d = b[lists]
        np.save(SAVE_DIR + '/target_data.npy'.format(i), c)
        np.save(SAVE_DIR + '/target_labels.npy'.format(i), d)

elif DATA_SET == 'svhn':
    a = np.load(os.path.join(DATA_DIR, 'train_data.npy'))
    b = np.load(os.path.join(DATA_DIR, 'train_labels.npy'))
    print(a.shape)
    for i in range(SHADOW_AMT):
        lists = rd.sample(range(73257), DATA_SIZE)
        c = a[lists]
        d = b[lists].reshape(-1)
        np.save(SAVE_DIR + '/shadow_data_{i}.npy'.format(i=i), c)
        np.save(SAVE_DIR + '/shadow_labels_{i}.npy'.format(i=i), d)

    a = np.load(os.path.join(DATA_DIR, 'test_data.npy'))
    b = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    b = b.reshape(-1)
    np.save(os.path.join(DATA_DIR, 'test_labels.npy'), b)
    for i in range(1):
        lists = list(range(0,1000,2))
        lists.extend(list(range(1000, DATA_SIZE + 500)))
        c = a[lists]
        d = b[lists]
        print(c.shape)
        np.save(SAVE_DIR + '/target_data.npy'.format(i), c)
        np.save(SAVE_DIR + '/target_labels.npy'.format(i), d)

elif DATA_SET == 'location':
    dataset = np.load('../shadow_dataset/location/data_complete.npz')
    x_data = dataset['x'][:, :]
    y_data = dataset['y'][:] - 1
    for i in range(SHADOW_AMT):
        lists = rd.sample(range(3500), DATA_SIZE)
        x_dat = x_data[lists]
        y_dat = y_data[lists]
        np.save(SAVE_DIR + '/shadow_data_{i}.npy'.format(i=i), x_dat)
        np.save(SAVE_DIR + '/shadow_labels_{i}.npy'.format(i=i), y_dat)
    
    lists = list(range(3500, 4500, 2))
    lists.extend(list(range(4500, 5000)))
    x_dat = x_data[lists]
    y_dat = y_data[lists]
    np.save(SAVE_DIR + '/target_data.npy', x_dat)
    np.save(SAVE_DIR + '/target_labels.npy', y_dat)

    lists = list(range(3500, 4500))
    x_dat = x_data[lists]
    y_dat = y_data[lists]
    np.save('../shadow_dataset/location/test_data.npy', x_dat)
    np.save('../shadow_dataset/location/test_labels.npy', y_dat)

elif DATA_SET == 'texas100':
    x_data = np.load('../shadow_dataset/texas100/train_data.npy')
    y_data = np.load('../shadow_dataset/texas100/train_labels.npy')
    for i in range(SHADOW_AMT):
        lists = rd.sample(range(50000), DATA_SIZE)
        x_dat = x_data[lists]
        y_dat = y_data[lists]
        np.save(SAVE_DIR + '/shadow_data_{i}.npy'.format(i=i), x_dat)
        np.save(SAVE_DIR + '/shadow_labels_{i}.npy'.format(i=i), y_dat)
    
    x_data = np.load('../shadow_dataset/texas100/test_data.npy')
    y_data = np.load('../shadow_dataset/texas100/test_labels.npy')
    lists = list(range(0, 1000, 2))
    lists.extend(list(range(1000, DATA_SIZE+500)))
    x_dat = x_data[lists]
    y_dat = y_data[lists]
    np.save(SAVE_DIR + '/target_data.npy', x_dat)
    np.save(SAVE_DIR + '/target_labels.npy', y_dat)

elif DATA_SET == 'purchase100':
    x_data = np.load('../shadow_dataset/purchase100/train_data.npy')
    y_data = np.load('../shadow_dataset/purchase100/train_labels.npy')
    for i in range(SHADOW_AMT):
        lists = rd.sample(range(150000), DATA_SIZE)
        x_dat = x_data[lists]
        y_dat = y_data[lists]
        np.save(SAVE_DIR + '/shadow_data_{i}.npy'.format(i=i), x_dat)
        np.save(SAVE_DIR + '/shadow_labels_{i}.npy'.format(i=i), y_dat)
    
    x_data = np.load('../shadow_dataset/purchase100/test_data.npy')
    y_data = np.load('../shadow_dataset/purchase100/test_labels.npy')
    lists = list(range(0, 1000, 2))
    lists.extend(list(range(1000, DATA_SIZE+500)))
    x_dat = x_data[lists]
    y_dat = y_data[lists]
    np.save(SAVE_DIR + '/target_data.npy', x_dat)
    np.save(SAVE_DIR + '/target_labels.npy', y_dat)



        
