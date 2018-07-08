#import
import numpy as np
import librosa
import pandas as pd
import ntpath
from glob import glob
from multiprocessing import Pool
from numpy import array_split
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import gc
import os
import re
from torchvision.transforms import Compose
from transforms_wav import *


class Meta:
    classes = ['background', 'bags', 'door', 'keyboard', 'knocking_door', 'ring', 'speech', 'tool']
    unknown = len(classes) + 1
    sampling_rate = 16000

class AudioSamplesDataset(Dataset):
    ONE_SECOND_LENGTH = Meta.sampling_rate
    
    def __init__(self, files, df, transforms=None):
        self.transforms = transforms
        self.df = df
        self.files = []
        self.labels = []
        for path in files:
            file_name = ntpath.basename(path)
            label = self.get_label(file_name)
            if label == Meta.unknown:
                continue
            self.labels.append(label)
            self.files.append(path)


    def get_label(self, id):
        selection = self.df[self.df['id'] == id]['label']
        if len(selection) == 0:
            return Meta.unknown
        return Meta.classes.index(selection.item())

    def __len__(self):
        return len(self.files)

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(Meta.classes)
        count = np.zeros(nclasses)
        for label in self.labels:
            count[label] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, label in enumerate(self.labels):
            weight[idx] = weight_per_class[label]
        return weight
    
    def pad_to_one_second(self, data):
        left_padding = (self.ONE_SECOND_LENGTH - data.shape[0]) // 2
        right_padding = self.ONE_SECOND_LENGTH - (left_padding + data.shape[0])
        return np.pad(data, (left_padding, right_padding), mode='constant')
    
    def get_random_one_second_sample(self, data):
        if len(data) <= self.ONE_SECOND_LENGTH:
            return self.pad_to_one_second(data)
        else:
            offset = np.random.randint(len(data) - self.ONE_SECOND_LENGTH)
            limit = offset + self.ONE_SECOND_LENGTH
            return data[offset:limit]

    def __getitem__(self, idx):
        path = self.files[idx]
        data = librosa.core.load(path, sr=Meta.sampling_rate, mono=True)[0]
        data = self.get_random_one_second_sample(data)
        sample = {'input': data, 'target': self.labels[idx], 'sample_rate': Meta.sampling_rate}
        if self.transforms is not None:
            sample = self.transforms(sample)
        data = torch.from_numpy(sample['input']).float()
        data = data.view(1, data.size(0), data.size(1))
        sample['input'] = data
        return sample


class FullClipDataset(Dataset):
    ONE_SECOND_LENGTH = Meta.sampling_rate

    def __init__(self, files, df, skip_unknonw=True, transforms=None):
        self.transforms = transforms
        self.df = df
        self.files = []
        self.labels = []
        for path in files:
            file_name = ntpath.basename(path)
            label = self.get_label(file_name)
            if skip_unknonw and label == Meta.unknown:
                continue
            self.labels.append(label)
            self.files.append(path)

    def get_label(self, id):
        selection = self.df[self.df['id'] == id]['label']
        if len(selection) == 0:
            return Meta.unknown
        return Meta.classes.index(selection.item())

    def __len__(self):
        return len(self.files)

    def pad_to_one_second(self, data):
        left_padding = (self.ONE_SECOND_LENGTH - data.shape[0]) // 2
        right_padding = self.ONE_SECOND_LENGTH - (left_padding + data.shape[0])
        return np.pad(data, (left_padding, right_padding), mode='constant')

    def get_random_one_second_sample(self, data):
        if len(data) <= self.ONE_SECOND_LENGTH:
            return self.pad_to_one_second(data)
        else:
            offset = np.random.randint(len(data) - self.ONE_SECOND_LENGTH)
            limit = offset + self.ONE_SECOND_LENGTH
            return data[offset:limit]

    def split_by_one_second(self, data):
        one_second_splits = np.split(data, np.arange(self.ONE_SECOND_LENGTH, len(data), self.ONE_SECOND_LENGTH))
        splits = []
        for split in one_second_splits:
            if split.shape[0] < self.ONE_SECOND_LENGTH:
                split = self.pad_to_one_second(split)
            splits.append(split)
        return splits

    def preprocess_splits(self, splits):
        preprocessed = []
        for data in splits:
            if self.transforms is not None:
                data = self.transforms(data)
            data = torch.from_numpy(data).float()
            data = data.view(1, data.size(0), data.size(1))
            preprocessed.append(data)
        return torch.stack(preprocessed)

    def __getitem__(self, idx):
        path = self.files[idx]
        file_name = ntpath.basename(path)
        data = librosa.core.load(path, sr=Meta.sampling_rate, mono=True)[0]
        splits = self.split_by_one_second(data)
        splits = self.preprocess_splits(splits)
        return {'input': splits, 'target': torch.tensor(self.labels[idx]), 'name': file_name}

