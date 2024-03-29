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
