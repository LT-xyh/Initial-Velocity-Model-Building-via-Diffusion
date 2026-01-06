import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.math_f import closest_power_of_two
from utils.modules import ResBlock


class MyAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=16, in_size=(70, 70), base_channel=64):
        super().__init__()
        x_size = max(in_size)
        base_size = closest_power_of_two(x_size)
        self.encoder = ResNetEncoder(in_channels=in_channels, in_size=x_size, base_size=base_size,
                                     base_channel=base_channel, channels=(2, 4), latent_channels=latent_channels)
        self.decoder = ResNetDecoder(latent_channels=latent_channels, out_channels=1, out_size=in_size,
                                     base_channel=base_channel, channels=(2, 4))

    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, cond=None):
        x = self.encode(x)
        x = self.decode(x)
        return x


class MyConditionalAE(MyAE):
    def __init__(self, in_channels=1, latent_channels=16, in_size=(70, 70), base_channel=64, cond_channels=64):
        super().__init__(in_channels, latent_channels, in_size, base_channel)
        self.decoder = ResNetDecoder(latent_channels=latent_channels + cond_channels, out_channels=1, out_size=in_size,
                                     base_channel=base_channel, channels=(2, 4))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, cond=None):
        return self.decoder(torch.cat([x, cond], dim=1))

    def forward(self, x, cond=None):
        x = self.encode(x)
        x = self.decode(x, cond)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1, in_size=70, base_size=64, base_channel=32, channels=(2, 4), latent_channels=16):
        super().__init__()
        self.in_size = in_size
        self.base_size = base_size
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=base_channel, kernel_size=7,
                                   padding=3)  # 7*7卷积层
        self.blocks = nn.ModuleList()
        for i in range(len(channels)):
            if i == 0:
                self.blocks.append(ResBlock(base_channel, base_channel * channels[i], stride=2))
            else:
                self.blocks.append(ResBlock(base_channel * channels[i - 1], base_channel * channels[i], stride=2))
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channel * channels[-1], out_channels=latent_channels, kernel_size=1), nn.Tanh())

    def forward(self, x):
        x = self.init_conv(x)
        x = F.interpolate(x, size=(self.base_size, self.base_size), mode='bilinear', align_corners=False)  # 70 -> 64
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, latent_channels=16, out_channels=1, out_size=(70, 70), base_channel=32, channels=(2, 4)):
        super().__init__()
        self.out_size = out_size
        self.init_conv = nn.Conv2d(in_channels=latent_channels, out_channels=base_channel, kernel_size=3, stride=1)
        self.blocks = nn.ModuleList()
        for i in range(len(channels)):
            if i == 0:
                self.blocks.append(nn.Sequential(ResBlock(base_channel, base_channel * channels[i], stride=1),
                                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ))
            else:
                self.blocks.append(
                    nn.Sequential(ResBlock(base_channel * channels[i - 1], base_channel * channels[i], stride=1),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ))
        self.final_conv = nn.Sequential(nn.Conv2d(base_channel * channels[-1], out_channels, 1), nn.Tanh(), )

    def forward(self, x):
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    def test1():
        x = torch.randn(4, 1, 70, 70)
        cond = torch.randn(4, 64, 16, 16)
        model = MyConditionalAE(latent_channels=16, cond_channels=64)
        out = model(x, cond)
        print(out.shape)


    test1()
