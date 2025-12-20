import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResBlock(nn.Module):
    """
    ResNet block for discriminator
    """
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       downsample: bool=True):
        super().__init__()
        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        if downsample or in_channels != out_channels:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, padding=0))
        else:
            self.conv_sc = None
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor):
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        if self.conv_sc is not None:
            x = self.conv_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return h + x



class Discriminator(nn.Module):
    """
    ResNet-based discriminator
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.block1 = ResBlock(in_channels, 64, downsample=True)
        self.block2 = ResBlock(64, 128, downsample=True)
        self.block3 = ResBlock(128, 256, downsample=True)
        self.block4 = ResBlock(256, 512, downsample=True)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.linear = spectral_norm(nn.Linear(512 * 8 * 8, 1))

    def forward(self, x: torch.Tensor):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = h.view(h.size(0), -1)
        out = self.linear(h)
        return out