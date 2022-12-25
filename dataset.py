import os
import json
from itertools import chain
import torch
import torchaudio
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm, trange


import wav2mel


class WMDataset(Dataset):
    def __init__(self, path2wavs='./wavs', random_crop=True, fragment_size=8192, melspec_config=None):
        self.random_crop = random_crop
        self.fragment_size = fragment_size
        self.path2wavs = path2wavs
        self.files = list(os.walk(path2wavs))[0][2]
        self.length_dataset = len(self.files)
        melspectrogrammator_config = melspec_config if melspec_config else wav2mel.MelSpectrogramConfig()
        self.melspectrogrammator = wav2mel.MelSpectrogram(melspectrogrammator_config)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        filename = self.path2wavs + self.files[idx]
        audio, sr = torchaudio.load(filename)
        #audio = audio.sum(0)
        start = 0
        if audio.size(1) > self.fragment_size:
            start = torch.randint(audio.size(1) - self.fragment_size, (1,))
        audio = audio[:,start:start+self.fragment_size]
        mel = self.melspectrogrammator(audio)
        return {'audio': audio, 'mel':mel, 'pad':self.melspectrogrammator.config.pad_value}

def collate_fn(dataset_items):
    batch_size = len(dataset_items)
    
    wavs = [item['audio'] for item in dataset_items]
    wav_lengths = torch.tensor([wav.size(-1) for wav in wavs]).long()

    mels = [item['mel'] for item in dataset_items]
    mel_lengths = torch.tensor([mel.shape[-1] for mel in mels]).long()

    batch_wavs = torch.zeros(batch_size, max(wav_lengths))
    batch_mels = torch.zeros(batch_size, mels[0].shape[-2], max(mel_lengths))

    for i, l in enumerate(wav_lengths):
        batch_wavs[i, :l] = wavs[i]

    for i, l in enumerate(mel_lengths):
        batch_mels[i, :, :l] = mels[i][0,:,:]


    result_batch = {'audio': batch_wavs,
                    'audio_length': wav_lengths,
                    'mel': batch_mels,
                    'mel_length': mel_lengths,}
    return result_batch