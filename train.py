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

gen.train()
mpd.train()
msd.train()

opt_g = torch.optim.AdamW(gen.parameters(), params['learning_rate'], betas=[params['adam_b1'], params['adam_b2']])
opt_d = torch.optim.AdamW(chain(mpd.parameters(), msd.parameters()), params['learning_rate'], betas=[params['adam_b1'], params['adam_b2']])

scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=params["lr_decay"], last_epoch=-1)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=params["lr_decay"], last_epoch=-1)

import wandb
wandb.init(project="GAN_vocoder_project")


for epoch in trange(20):
    for batch in tqdm(training_loader):
        audio = batch['audio'].to(device)
        mel = batch['mel'].to(device)

        audio_pred = gen(mel)
        mel_pred = melspectrogrammator_cuda(audio_pred.squeeze(1))

        opt_g.zero_grad()
        y_pdisc_real, y_pdisc_generated, _, _ = mpd(audio.unsqueeze(1), audio_pred.detach())
        loss_pdisc, losses_pdisc_real, losses_pdisc_generated = discriminator_loss(y_pdisc_real, y_pdisc_generated)

        y_scdisc_real, y_scdisc_generated, _, _ = msd(audio.unsqueeze(1), audio_pred.detach())
        loss_scdisc, losses_scdisc_real, losses_scdisc_generated = discriminator_loss(y_scdisc_real, y_scdisc_generated)

        loss_disc_all = loss_scdisc + loss_pdisc

        wandb.log({'Period discriminator loss': loss_pdisc.item()})
        wandb.log({'Scale discriminator loss': loss_scdisc.item()})

        loss_disc_all.backward()
        opt_d.step()

        opt_g.zero_grad()

        loss_mel = F.l1_loss(mel, mel_pred) * 45

        y_pdisc_real, y_pdisc_generated, featmap_p_real, featmap_p_generated = mpd(audio.unsqueeze(1), audio_pred)
        y_scdisc_real, y_scdisc_generated, featmap_sc_real, featmap_sc_generated = msd(audio.unsqueeze(1), audio_pred)
        loss_featmap_p = feature_loss(featmap_p_real, featmap_p_generated)
        loss_featmap_sc = feature_loss(featmap_sc_real, featmap_sc_generated)
        loss_gen_p, losses_gen_p = generator_loss(y_pdisc_generated)
        loss_gen_sc, losses_gen_sc = generator_loss(y_scdisc_generated)
        loss_gen_all = loss_gen_sc + loss_gen_p + loss_featmap_sc + loss_featmap_p + loss_mel

        wandb.log({'Generator scale loss': loss_gen_sc.item()})
        wandb.log({'Generator period loss': loss_gen_p.item()})
        wandb.log({'Generator scale feature loss': loss_featmap_sc.item()})
        wandb.log({'Generator period feature loss': loss_featmap_p.item()})
        wandb.log({'Generator mel loss': loss_mel.item()})

        loss_gen_all.backward()
        opt_g.step()
    scheduler_g.step()
    scheduler_d.step()
    
    if epoch%5 == 4:
        torch.save(gen.state_dict(), f'./weights/run/gen_epoch_{epoch}')

        torch.save({'gen': gen.state_dict(),
                    'mpd': mpd.state_dict(),
                    'msd': msd.state_dict(),
                    'opt_g': opt_g.state_dict(),
                    'opt_d': opt_d.state_dict(),
                    'config': params}, f'./weights/run/ckpt_epoch_{epoch}')