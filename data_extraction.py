import numpy as np
import pandas as pd
import librosa
import glob
import os
import matplotlib.pyplot as plt
import random
import pickle
# !pip install PyDub
# from pydub import AudioSegment
from sklearn import preprocessing

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

import librosa
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def remove_files(folder_path):
  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # print("Removing...",file_path)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # This will delete the file or link
        elif os.path.isdir(file_path):
            # If you also want to remove directories, uncomment the next line
            # shutil.rmtree(file_path)
            pass  # Currently, it does nothing with directories
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

def train_validate_test_split(df, train_percent=.5, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]].reset_index(drop=True)
    validate = df.iloc[perm[train_end:validate_end]].reset_index(drop=True)
    test = df.iloc[perm[validate_end:]].reset_index(drop=True)
    return train, validate, test

def rename_songs(df, split='train'):
    file_names = list()
    for index, row in df.iterrows():
        file_names.append(split + '_song_'+str(index))
    # df = pd.DataFrame({row['file_name'],row['label']})
    df['file_name'] = file_names
    return df

def get_feature():
  pass

# def extend_or_cut_song(song, length):
#     current_length = song.shape[1]
#     if current_length < length:
#         # Calculate the required padding
#         pad_length = length - current_length    
#         # Pad with zeros
#         song_padded = np.pad(song, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
#     else:
#     # No padding needed, but let's trim it just in case it's longer
#         song_padded = song[:, :length]
#     return torch.tensor(song_padded,dtype=torch.float32)

# Take 10sec snippets.
def take_ns_snippets(song, sr, chunk_len_s=10):
    samples_per_10_sec = 10 * sr
    channels, total_samples = song.shape
    num_full_snippets = total_samples // samples_per_10_sec
    snippets = []
    
    for i in range(num_full_snippets):
        start_sample = i * samples_per_10_sec
        end_sample = start_sample + samples_per_10_sec
        snippet = song[:, start_sample:end_sample]
        snippets.append(snippet)
    
    return snippets

def store_snipped_data(df, folder_path, split, features):
    # snip_df = pd.DataFrame()
    data_split_path = os.path.join(folder_path,split)
    remove_files(data_split_path)
    total_snippets = 0
    pbar = tqdm(total=len(df.index))
    for index, row in df.iterrows():
        # print(row['file_name'],row['label'])
        song_name = split + '_song_' + str(index)
        # print('changed_name: '+ song_name)
        print("Song Name: ",row['file_name'])
        # print("Label: ",row['label'])
        song, sr = torchaudio.load(row['file_name'])
        # print("SAMPLE_RATE: ",sr)
        snippets = take_ns_snippets(song, sr, chunk_len_s=10)

        # print("sampling rate:",sr)
        total_snippets += len(snippets)
        # print("Number of snippets:",total_snippets)
        for id, snip in tqdm(enumerate(snippets)):
            # Make all the snippets same size/Discard < 10sec snippets
            # print("snippet Length:",snip.shape[1])
            snip_song_name = song_name + '__snip_' + str(id) +'__'+ str(row['label']) + '.wav'
            print(snip_song_name)
            # print(snip)
            file_path = os.path.join(data_split_path,snip_song_name)
            # print(file_path)
            torchaudio.save(file_path, snip, sample_rate=sr, format='wav')
            # Save snippets
        pbar.update(index)
    print("{split} Snippets: ", total_snippets)

def prepare_model_input(folder_path,split,feature, save=False):
    data_path = os.path.join(folder_path,split)
    songs = glob.glob(os.path.join(data_path,'*.wav'))
    #  Convert snips to tensor, label them, label with snip name as well.[tensor,label,song_name]
    # Make a pickle file and save it in same folder path if save = True
    dataset = []
    for s in songs:
        # print(s)
        if feature == 'raw_waveform':
            tens_wave, sr = torchaudio.load(s)
            file_name = s.split('.')[0].split('/')[-1]
            label = int(file_name.split('__')[-1])
            dataset.append([tens_wave, (torch.tensor([label]),
                                                    file_name)])
            # print(label,file_name)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    if save == True:
        # print(folder_path)
        print(folder_path +f'_{feature}_{split}.pkl')
        pickle.dump(dataset,open(os.path.join(folder_path, f'{feature}_{split}.pkl'),'wb'))
    
    return dataset


###############################################################################################

# pwd = os.path.curdir
features = ['raw_waveform']#,'mfcc','cqcc','spectogram']
pwd = os.path.abspath(os.path.curdir)
print(pwd)
project_path = pwd #os.path.join(pwd,'drive', 'MyDrive', 'Prog_Rock_Project')
dataset_dir_path = os.path.join(project_path,'Dataset')
prog_rock_path = os.path.join(dataset_dir_path,'Progressive_Rock_Songs')
non_prog_rock_other_path = os.path.join(dataset_dir_path, 'Not_Progressive_Rock','Other_Songs')
non_prog_rock_pop_path = os.path.join(dataset_dir_path,'Not_Progressive_Rock','Top_Of_The_Pops')
drop_path = os.path.join(project_path, 'labeled_snip_dataset')
# os.mkdir(drop_path)
os.makedirs(drop_path,exist_ok=True)


# Iterate over all feature and generate train,test,valid folder.
for feature in features:
    feature_path = os.path.join(drop_path,feature+f'_features')

################### NEED TO delete files if we want to UPDATE dataset ##########################
os.makedirs(feature_path,exist_ok=True)
for s in ['train','test','valid']:
    os.makedirs(os.path.join(feature_path,s),exist_ok=True)

feature_dataset_list = [x for x in os.listdir(drop_path)]
print(feature_dataset_list)

# train_drop_path = os.path.join(raw_waveform_feature,'train')
# test_drop_path = os.path.join(raw_waveform_feature,'test')
# valid_drop_path = os.path.join(raw_waveform_feature,'valid')


# Read all the songs
prog_rock_files_list = glob.glob(os.path.join(prog_rock_path,'*.mp3'))
non_prog_rock_other_files_list = glob.glob(os.path.join(non_prog_rock_other_path,'*.mp3'))
non_prog_rock_pop_files_list = glob.glob(os.path.join(non_prog_rock_pop_path,'*.mp3'))
non_prog_rock_files_list = non_prog_rock_other_files_list + non_prog_rock_pop_files_list

prog_rock_label = [1 for s in prog_rock_files_list]
non_prog_rock_label = [0 for s in non_prog_rock_files_list]
df = pd.DataFrame(columns=['file_name','label'])
df['file_name'] = prog_rock_files_list + non_prog_rock_files_list
df['label'] = prog_rock_label + non_prog_rock_label

# print(df)
df = df.sample(frac=1).reset_index(drop=True)
train_split, valid_split, test_split = train_validate_test_split(df, train_percent=0.7,validate_percent=0.15,seed=None)

# print(train_split)
# file_names = df['file_name']
# file_names = [file_name.split(dir_path)[1].split('.')[0] for file_name in file_names]

print(df)

# #Snipiffy
print("Snippifyy.....")
df_train = train_split
df_test = test_split
df_valid = valid_split

# Store snippets
print(df_valid)
feature_path_list = [os.path.join(drop_path,f+'_features') for f in features]
for f in features:
    feature_path = os.path.join(drop_path,f+'_features')
    print(feature_path)
    print("Snip Train Data")
    snip_df_train = store_snipped_data(df_train, feature_path, split='train',features=f)
    print("Snip Valid Data")
    snip_df_valid = store_snipped_data(df_valid, feature_path,split='valid',features=f)
    print("Snip Test Data")
    snip_df_test = store_snipped_data(df_test, feature_path,split='test',features=f)


# Convert all the songs to model input

for f in features:
    fpath = os.path.join(drop_path,f+'_features')
    for split in ['train','test','valid']:
        dataset = prepare_model_input(fpath, split, feature=f,save=True)