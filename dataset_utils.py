"""
This file used to deal with dataset.
"""
import h5py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import copy


class BindingDataset(Dataset):
    def __init__(self, data_dir, name, is_single=True, train=True, ver=False, is_ver=False, train_ver_splie_rate=0.9, transforms=None, target_transforms=None):
        """
        Init torch dataset.
        data_dir: directory to the dataset h5 files.
        name: name of the data, eg: bars.
        is_single: whether the image contains only one object. (Default True)
        train: load train dataset or val. (Default True)

        split_train_ver: whether split train verification test. (Default False)
        is_val: this is verification test. (default False)
        train_ver_splie_rate: split rate of train and verification test. (Default 0.9)
        """
        super(BindingDataset, self).__init__()

        self.transforms = transforms
        self.target_transforms = target_transforms

        train_single_data, train_multi_data, test_data, train_single_label, train_multi_label, test_label = gain_dataset(data_dir, name)
        if train:
            if is_single:
                self.data = train_single_data
                self.label = train_single_label
            else:
                self.data = train_multi_data
                self.label = train_multi_label
        else:
            self.data = test_data
            self.label = test_label

        
        if train and ver:
            data_size = int(self.data.shape[0] * train_ver_splie_rate)
            if is_ver:
                self.data = self.data[data_size:, :]
                self.label = self.label[data_size:, :]
            else:
                self.data = self.data[:data_size, :] 
                self.label = self.label[:data_size, :]
    
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(dim=1)
        self.label = torch.tensor(self.label, dtype=torch.float32).unsqueeze(dim=1)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.label[idx]
        if self.transforms:
            img = self.transforms(img)
        if self.target_transforms:
            label = self.target_transforms(label)
        return img, label 

    def __len__(self):
        return self.data.shape[0]


class BindingDatasetThread(Dataset):
    def __init__(self, data, label):
        super(BindingDatasetThread, self).__init__()
        self.data = copy.deepcopy(data)
        self.label = copy.deepcopy(label)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.label[idx]
        return img, label 

    def __len__(self):
        return self.data.shape[0]


def open_dataset(data_dir, name):
    """
    open dataset files.
    data_dir: directory to the dataset h5 files
    name: name of the data, eg: bars
    """
    filename = os.path.join(data_dir, name + '.h5')
    return h5py.File(filename, 'r')


def gain_dataset(data_dir, name):
    with open_dataset(data_dir, name) as f:
        train_multi_data = f["train_multi"]["default"][:]
        train_single_data = f["train_single"]["default"][:]
        test_data = f["test"]["default"][:]

        train_multi_label = f["train_multi"]["groups"][:]
        train_single_label = f["train_single"]["groups"][:]
        test_label = f["test"]["groups"][:]

    return train_single_data, train_multi_data, test_data, train_single_label, train_multi_label, test_label


def gain_verf_dataset(data_dir, name):
    if name == "bars":
        with open_dataset(data_dir, name) as f:
            hori = f["train_single_hori"]["default"][:]
            vert = f["train_single_vert"]["default"][:]
        return hori, vert
    elif name == "shapes":
        with open_dataset(data_dir, name) as f:
            s0 = f["train_single_0"]["default"][:]
            s1 = f["train_single_1"]["default"][:]
            s2 = f["train_single_2"]["default"][:]
        return s0, s1, s2


def test_bars_dataset():
    with open_dataset("./tmp_data", "bars") as f:
        print(f["train_multi"]["default"][:, 0])
        print(f["train_multi"]["groups"][:, 0])
    bars_dataset = BindingDataset("./tmp_data", "bars", is_single=False, train=True, ver=True, is_ver=False)
    train_loader = DataLoader(dataset=bars_dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(32):
        for iter, data in enumerate(train_loader):
            img, label = data
            print("img shape", img.shape)
            print(label.shape)
            break
        break


def test_corners_dataset():
    with open_dataset("./tmp_data", "corners") as f:
        print(f["train_multi"]["default"][:, 0])
        print(f["train_multi"]["groups"][:, 0])
    corners_dataset = BindingDataset("./tmp_data", "corners", is_single=False, train=True, ver=True, is_ver=False)
    train_loader = DataLoader(dataset=corners_dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(32):
        for iter, data in enumerate(train_loader):
            img, label = data
            print(img.shape)
            print(label.shape)
            break
        break


def test_verf(data_name):
    if data_name == "bars":
        hori, vert = gain_verf_dataset("./tmp_data", data_name)
        print(hori[0], vert[0])
    if data_name == "shapes":
        s0, s1, s2 = gain_verf_dataset("./tmp_data", data_name)
        print(s0[0], s1[0], s2[0])

