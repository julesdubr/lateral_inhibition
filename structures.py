import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LateralInhibition(nn.Module):
    __constants__ = ["channel_in"]

    def __init__(self, channel_in=3, kernel_size=3, weights="zeros"):
        super(LateralInhibition, self).__init__()

        self.channel_in = channel_in

        self.G = self.gaussian_filter(kernel_size)
        self.C = self.G[0, 0, 1, 1].item()

        # Amplitude weights, can be zero, positive, or negative (ie. no LI, LI, -LI)
        self.weights = nn.Parameter(getattr(torch, weights)(1, channel_in, 1, 1))

        # Additional parameters: factor (v), shift (m), and bias (b).
        self.m = nn.Parameter(torch.zeros(1))
        self.v = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Lateral interaction design
        (batch, C, H, W) = x.shape
        li = F.conv2d(x.view(-1, 1, H, W), self.G, padding=1).view(batch, C, H, W)
        li = (li - self.C) * self.weights

        # Applying lateral inhibition
        out = x - li
        out = (out + self.m) * self.v + self.b
        return out

    # 2D Gaussian low-pass filter
    def gaussian_filter(self, kernel_size, sigma=1):

        # Initializing value of x,y as grid of kernel size in the range of kernel size
        u, v = np.meshgrid(
            np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size)
        )

        # lower normal part of gaussian
        normal = 1 / (2 * np.pi * sigma**2)

        # Calculating Gaussian filter
        G = np.exp(-(u**2 + v**2 / (2.0 * sigma**2))) * normal
        return torch.FloatTensor(G).repeat(1, 1, 1, 1).to(device)


class LIBlock(nn.Module):
    def __init__(self, block: BasicBlock):
        super(LIBlock, self).__init__()

        self.block = block
        self.li = LateralInhibition(self.block.conv1.in_channels)

    def forward(self, x):
        identity = x

        out = self.li(x)  # added lateral inhibition
        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        return out
