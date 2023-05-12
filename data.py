import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import numpy as np
from preprocess import wav_to_fbank, TacotronSTFT

def collate(batch):
    audios = torch.stack([item[0] for item in batch])
    metadatas = [item[1] for item in batch]
    return audios, metadatas

def get_all_mp3_files(path):
    '''Get all the mp3 files in the path recursively, useful for labelled datasets'''
    mp3_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

class FMADataset(Dataset):
    def __init__(self, directory, target_sample_rate=16000, segment_length=10, hop_len = 160, n_windows = None):
        self.directory = directory
        self.target_sample_rate = target_sample_rate
        self.segment_length = segment_length
        self.file_list = get_all_mp3_files(directory)
        if n_windows is not None:
            self.segment_windows = n_windows
        else:
            segment_windows = self.target_sample_rate * self.segment_length // hop_len
            self.segment_windows = 2 ** (int(np.log2(segment_windows)) + 1)
        self.segment_samples = (self.segment_windows - 1) * hop_len

        with open("preprocess_config.json", "r") as f:
            preprocess_config = json.load(f)
        self.transforms = TacotronSTFT(**preprocess_config)

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, index):
        file_path = self.file_list[index]
        if index == len(self.file_list) - 1:
            index = 0
        try:
            mels, log_mag_stft, waveform = wav_to_fbank(file_path, self.segment_windows, self.transforms)
        except:
            return self.__getitem__(index + 1)
        if mels == None:
            return self.__getitem__(index + 1)
        return waveform, mels
   
class WaveDataset(Dataset):
    def __init__(self, directory, target_sample_rate=16000, segment_length=10, hop_len = 160, n_windows = None):
        self.directory = directory
        self.target_sample_rate = target_sample_rate
        self.segment_length = segment_length
        self.file_list = [file for file in os.listdir(directory) if file.endswith('.wav')]
        if n_windows is not None:
            self.segment_windows = n_windows
        else:
            segment_windows = self.target_sample_rate * self.segment_length // hop_len
            self.segment_windows = 2 ** (int(np.log2(segment_windows)) + 1)
        self.segment_samples = (self.segment_windows - 1) * hop_len
        with open("preprocess_config.json", "r") as f:
            preprocess_config = json.load(f)
        self.transforms = TacotronSTFT(**preprocess_config)

    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, index):
        file_path = os.path.join(self.directory, self.file_list[index])
        mels, log_mag_stft, waveform = wav_to_fbank(file_path, self.segment_windows, self.transforms)
        if mels == None:
            return self.__getitem__(index + 1)
        return waveform, mels
    
    def __getit2em__(self, index):
        file_path = os.path.join(self.directory, self.file_list[index])
       # waveform = read_wav_file(file_path, segment_length = self.segment_samples)
        print(waveform.shape)
        target_length = self.segment_samples
        waveform_length = waveform.shape[1]

        # turn to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim = 0).squeeze(0)
        else:
            waveform = waveform.squeeze(0)
      #  mels = 
        # constant length output for equal size mels
        if waveform_length < target_length:
            padding = torch.zeros((target_length - waveform_length))
            waveform = torch.cat((waveform, padding), dim = 0)
        elif waveform_length > target_length:
            waveform = waveform[:target_length]
        return waveform

class NSynthDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(os.path.join(root_dir, metadata_file), 'r') as f:
            self.metadata = json.load(f)

        self.file_names = list(self.metadata.keys())

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_names[idx]
        wav_path = os.path.join(self.root_dir, 'audio', f'{file_name}.wav')
        waveform, _ = torchaudio.load(wav_path)
        waveform = torch.cat([waveform, torch.zeros(waveform.shape[0], 1024)], dim = 1)

        example_metadata = self.metadata[file_name]

        if self.transform:
            waveform = self.transform(waveform)
            
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim = 0)
        else:
            waveform = waveform.reshape(-1)
        return waveform, example_metadata

def load_data(root_dir, batch_size = 4, workers = 4, shuffle = True):
    print("dataloader with ", workers, " workers")
    dataset = FMADataset(root_dir)
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = workers, shuffle = shuffle)
    return loader