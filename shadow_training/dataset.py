""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.label = labels
        self.transform = transform
        r = self.data[:, :1024].reshape(-1, 32, 32)
        g = self.data[:, 1024:2048].reshape(-1, 32, 32)
        b = self.data[:, 2048:].reshape(-1, 32, 32)
        self.image = np.dstack((r, g, b))
        self.image = self.image.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.label[index]
        image = self.transform(self.image[index])
        return image, label


class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.label = labels
        self.transform = transform
        r = self.data[:, :1024].reshape(-1, 32, 32)
        g = self.data[:, 1024:2048].reshape(-1, 32, 32)
        b = self.data[:, 2048:].reshape(-1, 32, 32)
        self.image = np.dstack((r, g, b))
        self.image = self.image.reshape((-1, 32, 3, 32)).transpose(0, 1, 3, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.label[index]
        image = self.transform(self.image[index])
        return image, label
