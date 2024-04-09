import os
import pickle as pickle


from multiprocessing import Process
from subprocess import call

from dataclasses import dataclass


import numpy as np
from scipy.io.wavfile import read

from torch import nn
from torch.nn import functional as F

<<<<<<< HEAD
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


=======
import torch
import pt_util


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.best_accuracy = 0

    # Saves the current model
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.save_model(file_path, num_to_keep)
            self.best_accuracy = accuracy

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)
    def loss(self, prediction, label, reduction='mean'):
        return nn.BCEWithLogitsLoss(reduction='mean')(prediction, label)


class TwoDCNN(Model):
    def __init__(self, num_classes=1):
        super(TwoDCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        # Remove dynamic calculation for simplicity in this example
        # Assuming fixed input size, calculate _to_linear based on the architecture and input dimensions
        # This calculation needs to be adjusted based on the actual size after the pooling layers
        # self._to_linear = 128 * 32 * 54  
        
        self.fc1 = nn.Linear(53248, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the tensor while keeping the batch dimension
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class OneDCNN(Model):
    def __init__(self, input_channels=2, input_length=441000):
        super(OneDCNN, self).__init__()
        

        self.conv1 = nn.Conv1d(in_channels=2,out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool1d(4, stride=4)
        

        self._to_linear = 0
        self.conv_output_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)  # Dynamically set based on conv_output_size calculation
        self.fc2 = nn.Linear(256, 64)
        
        # Output layer for binary classification
        self.output = nn.Linear(64, 1)
        
    def conv_output_size(self, L_in, kernel_size=3, stride=1, padding=1, pool_kernel_size=4, pool_stride=4):

        def calc_output_length(L_in, kernel_size, stride, padding, dilation=1):
            return ((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) // stride) + 1
        

        L_out = calc_output_length(L_in, kernel_size, stride, padding)
        L_out = calc_output_length(L_out, pool_kernel_size, pool_stride, 0) 
        L_out = calc_output_length(L_out, kernel_size, stride, padding)
        L_out = calc_output_length(L_out, pool_kernel_size, pool_stride, 0)
        L_out = calc_output_length(L_out, kernel_size, stride, padding)
        L_out = calc_output_length(L_out, pool_kernel_size, pool_stride, 0)

        self._to_linear = 64 * L_out

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self._to_linear)
        

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        

        x = self.output(x)
        return x

    def loss(self, prediction, label, reduction='mean'):
        return nn.BCEWithLogitsLoss(reduction='mean')(prediction, label)
>>>>>>> 1009a6e3d2972630230603b3c2c1d56709ad0f63




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
