from tqdm.notebook import trange, tqdm
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
from dataset import WMDataset, collate_fn
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, discriminator_loss, generator_loss



melspectrogrammator_config = wav2mel.MelSpectrogramConfig()
melspectrogrammator = wav2mel.MelSpectrogram(melspectrogrammator_config)
melspectrogrammator_cuda = wav2mel.MelSpectrogram(melspectrogrammator_config).cuda()

path2wavs = './wavs/'
random_crop = True
dataset = WMDataset(path2wavs=path2wavs, random_crop=random_crop, melspec_config=melspectrogrammator_config)

training_loader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=0
)


with open('./config_v2.json', 'r') as f:
    params = json.load(f)
device = 'cuda'

gen = Generator(**params).to(device)

mpd = MultiPeriodDiscriminator().to(device)
msd = MultiScaleDiscriminator().to(device)

gen.train(False)
mpd.train(False)
msd.train(False)

gen.load_state_dict(torch.load('./weights/try2/gen_epoch_14'))

for j in range(1,4):
    audio, sr = torchaudio.load(f'./test_wavs/audio_{j}.wav')
    mel_p = melspectrogrammator(audio)
    with torch.no_grad():
        synth = []
        for i in range(0, mel_p.size(-1), 32):
            synth.append(gen(mel_p[:,:,i:i+32].cuda()))
    result = torch.cat(synth, dim=-1)
    torchaudio.save(f'./test_wavs/res_{j}.wav', result[0].detach().cpu(), sr)