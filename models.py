from itertools import chain
import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


LRELU_SLOPE = 0.1

def calc_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=((kernel_size-1)*dilation[0])//2)),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=((kernel_size-1)*dilation[1])//2))
        ])
        for conv in self.convs:
            nn.init.normal_(conv.weight)

    def forward(self, x):
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x
            
class Generator(nn.Module):
    def __init__(self, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, resblock_dilation_sizes, **kwargs):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(80, upsample_initial_channel, 7, 1, padding=3))

        self.tconvs = nn.ModuleList()
        for i, (up_sz, up_ksz) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.tconvs.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                up_ksz, up_sz, padding=(up_ksz-up_sz)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.tconvs)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (r_kersz, r_dilsz) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, r_kersz, r_dilsz))
                
        for conv in self.tconvs:
            nn.init.normal_(conv.weight)

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        nn.init.normal_(self.conv_post.weight)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.tconvs[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        batch_sz, chan_sz, time_sz = x.shape
        if time_sz % self.period != 0: # pad first
            n_pad = self.period - (time_sz % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            time_sz = time_sz + n_pad
        x = x.view(batch_sz, chan_sz, time_sz // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=[2,3,5,7,11]):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([])
        for p in periods:
            self.discriminators.append(DiscriminatorP(p))

    def forward(self, y, y_hat):
        y_disc_reals = []
        y_disc_generateds = []
        fmap_reals = []
        fmap_gens = []
        for i, disc in enumerate(self.discriminators):
            y_disc_real, fmap_real = disc(y)
            y_disc_generated, fmap_gen = disc(y_hat)
            y_disc_reals.append(y_disc_real)
            fmap_reals.append(fmap_real)
            y_disc_generateds.append(y_disc_generated)
            fmap_gens.append(fmap_gen)

        return y_disc_reals, y_disc_generateds, fmap_reals, fmap_gens



class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_disc_reals = []
        y_disc_generateds = []
        fmap_reals = []
        fmap_gens = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_disc_real, fmap_real = disc(y)
            y_disc_generated, fmap_gen = disc(y_hat)
            y_disc_reals.append(y_disc_real)
            fmap_reals.append(fmap_real)
            y_disc_generateds.append(y_disc_generated)
            fmap_gens.append(fmap_gen)

        return y_disc_reals, y_disc_generateds, fmap_reals, fmap_gens




def feature_loss(fmap_real, fmap_gen):
    loss = 0
    for fmap_disc_real, fmap_disc_gen in zip(fmap_real, fmap_gen):
        for fmap_real, fmap_gen in zip(fmap_disc_real, fmap_disc_gen):
            loss += torch.mean(torch.abs(fmap_real - fmap_gen))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    real_losses = []
    gen_losses = []
    for disc_real, disc_gen in zip(disc_real_outputs, disc_generated_outputs):
        real_loss = torch.mean((1-disc_real)**2)
        gen_loss = torch.mean(disc_gen**2)
        loss += (real_loss + gen_loss)
        real_losses.append(real_loss.item())
        gen_losses.append(gen_loss.item())

    return loss, real_losses, gen_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for disc_gen in disc_outputs:
        l = torch.mean((1-disc_gen)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses