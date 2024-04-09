import os
import sys
import argparse
import os
import math
import pickle as pickle
import glob
import copy
import logging
import time
from typing import Any, Callable, Optional, List, Sequence

import multiprocessing
from multiprocessing import Process
from subprocess import call

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import numpy as np
from scipy.io.wavfile import read

import torch
import torch.utils.data
from torch import nn, Tensor
from torch.nn import functional as F
import torch.optim as optim

from torchaudio import functional
from torchaudio import transforms

import torchvision.datasets.utils
from torchvision.models._utils import _make_divisible


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, batch_size, data_type='train', label_index=0):
        super(MusicDataset, self).__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        pkl_list = glob.glob(os.path.join(data_dir,'*.pkl'))
        print(pkl_list)
        file_name = [file for file in pkl_list if data_type in file][0]
        print(file_name)
        data_pkl = open(file_name,'rb')
        self.data = pickle.load(data_pkl)
        data_pkl.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1][0]

parser = argparse.ArgumentParser()
parser.add_argument("-d","--data_path", help = "Path to model Inputs(pkl files)")
parser.add_argument("-l","--log_path",help="Path to save the logs and model")
args = parser.parse_args()

DATA_PATH = args.data_path
LOG_PATH =  args.log_path
BATCH_SIZE = 3
test_set = MusicDataset(DATA_PATH, BATCH_SIZE, data_type='test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=True, drop_last=False)
for idx, (data,label) in enumerate(test_loader):
    print(idx, data.shape, label.shape)




##################################################
class TrialCNN(nn.Module):
    def __init__(self):
        super(TrialCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,  padding=1),
            nn.BatchNorm2d(16),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,  padding=1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        #nn.ReLU(),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        #nn.ReLU(),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer5 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        #nn.ReLU(),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.fc1 = nn.Sequential(
            # nn.Linear(in_features=882048, out_features=100),
            nn.Linear(in_features=8832, out_features=1000),
            #nn.ReLU()
            nn.LeakyReLU(),
        )
        self.drop = nn.Dropout(0.25)

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=100),
            #nn.ReLU()
            nn.LeakyReLU(),
        )
        self.drop = nn.Dropout(0.25)

        self.fc3 = nn.Linear(in_features=100, out_features=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out
#################################################################
