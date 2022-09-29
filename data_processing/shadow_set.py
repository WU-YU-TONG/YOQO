import numpy as np
import random as rd

train_data = np.load('train_data.npy')
train_label = np.load('train_labels.npy')
print(train_label)

# targets_data = train_data[np.where(allocated == 0), :]
# targets_labels = train_label[np.where(allocated == 0)]
# print(targets_labels.shape)
# targets_data = targets_data.reshape(2383, 3072)
# np.save('../shadow_dataset/targets_data.npy', targets_data)
# np.save('../shadow_dataset/targets_labels.npy', targets_labels)


for iter in range(26):
    shadow_list = rd.sample(range(0, 50000), 2500)
    shadow_data = train_data[shadow_list, :]
    shadow_labels = train_label[shadow_list]
    np.save('../shadow_dataset/2500/shadow_data_{}.npy'.format(iter), shadow_data)
    np.save('../shadow_dataset/2500/shadow_labels_{}.npy'.format(iter), shadow_labels)
