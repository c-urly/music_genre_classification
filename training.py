import os
import sys
import argparse
import os
import math
import pickle as pickle
import glob
import copy
import json
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
from torchaudio import transforms as T
import torch.optim as optim

import torchaudio


import pt_util


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from model import TwoDCNN, OneDCNN


parser = argparse.ArgumentParser()
parser.add_argument("-d","--data_path", help = "Path to model Inputs(pkl files)")
parser.add_argument("-l","--log_path",help="Path to save the logs and model")
args = parser.parse_args()

MAX_LENGTH = 441000
num_channels = 2
# batch = 60


start_epoch = 0
feature_path = './labeled_snip_dataset/raw_waveform_features/'
BASE_PATH = feature_path
train_dir = os.path.join(feature_path,'train')
test_dir = os.path.join(feature_path,'test')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

def get_mfcc(waveform, sample_rate):
  n_fft = 2048
  win_length = None
  hop_length = 512
  n_mels = 256
  n_mfcc = 256

  mfcc_transform = T.MFCC(
      sample_rate=sample_rate,
      n_mfcc=n_mfcc,
      melkwargs={
          "n_fft": n_fft,
          "n_mels": n_mels,
          "hop_length": hop_length,
          "mel_scale": "htk",
      },
  )

  return mfcc_transform(waveform)

def get_specgram(waveform):
    spectrogram = T.Spectrogram(n_fft=512)
    return spectrogram(waveform)

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, feature='raw_waveform'):
        super(MusicDataset, self).__init__()

        self.data_dir = data_dir

        self.data_list = glob.glob(os.path.join(data_dir,'*.wav'))
        self.feature = feature

    def __len__(self):
        return len(self.data_list)

    def _get_feature_data(self, file_path, feature):
        song,sr = torchaudio.load(file_path)
        if feature == 'raw_waveform':
            return song
        if feature == 'mfcc':
            return get_mfcc(song,sr)
        if feature == 'specgram':
            return get_specgram(song)

    def __getitem__(self, index):
        file_path = self.data_list[index]

        label = int(os.path.basename(file_path).split('.')[0].split('__')[-1])
        feature_data = self._get_feature_data(file_path, self.feature)

        data = [feature_data, (torch.tensor([label]), file_path)]
        return data

def setup_logger(name_logfile, path_logfile):
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(path_logfile, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger

def train_binary(model, epoch, data_loader, device, optimizer,pbar, pbar_update, BATCH_SIZE):
    model.to(device)
    model.train()
    losses = []
    for batch_idx, (data, (target,file_name)) in enumerate(data_loader):
        # print(data, target)
        # print(f"Data shape: {data.shape}, Target shape: {target.shape}")
        data = data.to(device)
        target = target.float()
        target = target.to(device)
        output = model(data)
        # print(target.shape)
        # print(data.shape)
        # print(output.shape)
        loss = model.loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 20:
            print(f"Train Epoch: {epoch} [{(batch_idx+1) * BATCH_SIZE}/{len(data_loader) * BATCH_SIZE} ({100. * (batch_idx+1) / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        pbar.update(pbar_update)
        losses.append(loss.item())
    ave_losses = np.mean(losses)
    print(f"Train Epoch: {epoch} total average loss: {ave_losses:.6f}")
    return ave_losses

def update_classification_json(correctly_classified, misclassified):
    """
    Updates or creates a JSON file with correctly classified and misclassified filenames.

    Parameters:
    - file_path: Path to the JSON file.
    - data_path: The DATA_PATH key to update in the file.
    - correctly_classified: List of correctly classified file names.
    - misclassified: List of misclassified file names.
    """
    print(correctly_classified)
    print(misclassified)
    # Check if the file exists and read its content; if not, initialize an empty dict
    # if os.path.exists(file_path):
    #     with open(file_path, 'r') as file:
    #         try:
    #             data = json.load(file)
    #         except json.JSONDecodeError:
    #             data = {}
    # else:
    #     data = {}

    # # Update the data structure with new values, appending to existing lists if the key already exists
    # if data_path not in data:
    #     data[data_path] = {"correctly_classified": correctly_classified, "misclassified": misclassified}
    # else:
    #     data[data_path]["correctly_classified"].extend(correctly_classified)
    #     data[data_path]["misclassified"].extend(misclassified)

    # # Write the updated content back to the file
    # with open(file_path, 'w') as file:
    #     json.dump(data, file, indent=4)

def test_binary(model, epoch, data_loader, device, BATCH_SIZE):
    model.to(device)
    model.eval()
    correct = 0
    losses = []
    all_pred = []
    all_proba = []
    all_target = []
    all_mis_classified = []
    all_correctly_classified = []
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, (target,file_names)) in enumerate(data_loader):
            target = target.float()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            # print(target)
            # print(output)
            losses.append(model.loss(output, target).item())
            pred = (F.sigmoid(output) > 0.5).float()
            # print(pred)
            correct_mask = pred.eq(target.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            pred_cpu = pred.cpu().numpy()
            target_cpu = target.view_as(pred).cpu().numpy()
            for item, include in zip(file_names, correct_mask):
                if include:
                    all_correctly_classified.append(item)
                else:
                    all_mis_classified.append(item)     
            all_proba.append(output.cpu().numpy())
            all_pred.append(pred_cpu)
            all_target.append(target_cpu)
            if batch_idx % 20 == 0:
                print(f"Testing batch {batch_idx} of {len(data_loader)}")
                print(f"Correct count: {num_correct}")
    update_classification_json(all_correctly_classified, all_mis_classified)
    test_loss = np.mean(losses)
    # conf_matrix = confusion_matrix(np.concatenate(all_target, axis=0), np.concatenate(all_pred, axis=0))
    # print(f'Confusion matrix:\n{conf_matrix}')
    # class_report = classification_report(np.concatenate(all_target, axis=0), np.concatenate(all_pred, axis=0))
    # print(f'Classification Report:\n{class_report}')
    area_under_curv = roc_auc_score(np.concatenate(all_target, axis=0), np.concatenate(all_proba, axis=0).squeeze(), multi_class='ovr')
    # print(f'Area Under The Curve:\n{area_under_curv}')
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / (len(data_loader.dataset)):.2f}%). Average loss is: {test_loss:.6f}\n")
    test_accuracy = 100. * correct / len(data_loader.dataset)
    return test_loss, test_accuracy, area_under_curv

def train(BASE_PATH):
    BATCH_SIZE = 60
    learning_rate = 0.001
    num_epochs = 50
    WEIGHT_DECAY = 0.00005
    SCHEDULER_EPOCH_STEP = 4
    SCHEDULER_GAMMA = 0.8
    USE_CUDA = True
    PRINT_INTERVAL = 10
    VALID_BATCH_SIZE = 60
    num_workers = 10
    use_cuda = USE_CUDA and torch.cuda.is_available()
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    eval_area_under_curvs = []

    # model = AudioClassifier()
    model = TwoDCNN()
    model.to(device)
    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}
    train_set = MusicDataset(train_dir, feature='mfcc')
    test_set = MusicDataset(test_dir, feature='mfcc')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,**kwargs)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,**kwargs)


    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                            weight_decay= WEIGHT_DECAY, amsgrad=True)
    start_epoch = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_EPOCH_STEP, gamma=SCHEDULER_GAMMA)

    pbar_update = 1 / (len(train_loader))
    prev_learning_rate = learning_rate
    with tqdm(total=num_epochs) as pbar:
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                train_loss, eval_loss, eval_accuracy, eval_area_under_curv = 0, 0, 0, 0

                train_loss = train_binary(model, epoch, train_loader, device, optimizer, pbar, pbar_update, BATCH_SIZE)
                eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, epoch, test_loader, device, VALID_BATCH_SIZE)

                model.save_best_model(eval_accuracy, BASE_PATH + '/checkpoints/%03d.pt' % epoch, num_to_keep=5)
                scheduler.step()
                train_losses.append((epoch, train_loss))
                eval_losses.append((epoch, eval_loss))
                eval_accuracies.append((epoch, eval_accuracy))
                eval_area_under_curvs.append((epoch, eval_area_under_curv))

        except KeyboardInterrupt as ke:
            print('Interrupted')
        except:
            import traceback
            traceback.print_exc()
        finally:
            print('Saving final model')
            model.save_model(BASE_PATH + '/checkpoints/f%03d.pt' % epoch, num_to_keep=5)
            return model, device

def validate_a_classifier(BASE_PATH, device_instant, num_classes=1):    
    num_workers = 0
    EPOCHS = 5000
    BATCH_SIZE=60
    USE_CUDA = True
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.00005
    SCHEDULER_EPOCH_STEP = 4
    SCHEDULER_GAMMA = 0.8

    LOG_PATH = BASE_PATH + '/logs/log' + '1' + '.pkl'

    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = device_instant
    print(f'Using device {device}')
    
    print(f'num workers: {num_workers}')

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}
    model = TwoDCNN()
    print(f"Updated_model({0}): {model}")
    model.to(device)
    print(f"{model}")
    valid_set = MusicDataset(test_dir, feature='mfcc')
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,**kwargs)


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_EPOCH_STEP,
                                          gamma=SCHEDULER_GAMMA)
    start_epoch=0
    start_epoch = model.load_model(BASE_PATH+'/checkpoints/f050.pt')

    print(f"Updated_model({start_epoch}): {model}")
    print(f'Validation results:')
    # train_losses, eval_losses, eval_accuracies, eval_area_under_curvs = pt_util.read_log(LOG_PATH, ([], [], [], []))
    eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, start_epoch, valid_loader, device, BATCH_SIZE)

    return model, device

train(feature_path)


# validate_a_classifier(feature_path, device)
# for data, (label,file_name) in train_loader:
#     print(data.shape, label.shape)

# for data, (label,file_name) in test_loader:
#     print(data.shape, label.shape)