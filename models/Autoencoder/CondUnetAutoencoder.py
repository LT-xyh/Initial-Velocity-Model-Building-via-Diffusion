from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.Autoencoder.MyAE import MyAE
from models.conditional_encoder.CondFusionPyramid70 import ConvBNAct, ResBlock, CondFusionPyramid70


class CondUnetAutoencoder(MyAE):
    def __init__(self, in_channels=1, latent_channels=16, in_size=(70, 70), base_channel=64, ):
        super().__init__(in_channels, latent_channels, in_size, base_channel)
        self.decoder = ConditionalUNetDecoder(latent_channels, out_channels=1, final_act='tanh')

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, cond=None):
        return self.decoder(z=x, cond_pyr=cond)

    def forward(self, x, cond=None):
        x = self.encode(x)
        x = self.decode(x, cond)
        return x


# ====================== 条件 U-Net 解码器 ======================
class ConditionalUNetDecoder(nn.Module):
    """
    将 latent (B,16,16,16) 解码为 (B,1,70,70)，并在 16/32/64/70 四个尺度注入条件融合图。
    - 入口：latent_channels=16
    - 条件：通过 CondFusionPyramid70 提供 {'s16','s32','s64','s70'}
    - 结构：16→32→64→70，多尺度拼接 + 残差
    """

    def __init__(self, latent_channels: int = 16, base_ch: int = 64,
                 cond_C: Dict[str, int] = {'s16': 64, 's32': 64, 's64': 64, 's70': 32}, out_channels: int = 1,
                 final_act: str = "tanh"  # "tanh" | "none" | "sigmoid" | "softplus"
                 ):
        super().__init__()
        self.final_act = final_act

        # 16x16
        self.init_conv = ConvBNAct(latent_channels, base_ch, k=3)

        # 16 → (16-Block)
        self.dec16 = ResBlock(base_ch + cond_C['s16'], base_ch, d1=1, d2=1)

        # 16 → 32
        self.up32 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec32 = ResBlock(base_ch + cond_C['s32'], base_ch, d1=1, d2=1)

        # 32 → 64
        self.up64 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec64 = ResBlock(base_ch + cond_C['s64'], base_ch, d1=1, d2=1)

        # 64 → 70（直接插值到目标尺寸）
        self.dec70_in = ConvBNAct(base_ch, base_ch, k=3)
        self.dec70 = ResBlock(base_ch + cond_C['s70'], base_ch, d1=1, d2=1)

        # 输出头
        self.head = nn.Conv2d(base_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, z: torch.Tensor, cond_pyr: Dict[str, torch.Tensor]):
        """
        z: (B,16,16,16)
        cond_pyr: {'s16':(B,C16,16,16), 's32':(B,C32,32,32), 's64':(B,C64,64,64), 's70':(B,C70,70,70)}
        """
        B, C, H, W = z.shape
        assert (H, W) == (16, 16), "latent 期望为 16×16"

        # 16
        x = self.init_conv(z)  # (B,base,16,16)
        x = torch.cat([x, cond_pyr['s16']], dim=1)  # 拼接条件
        x = self.dec16(x)  # (B,base,16,16)

        # 32
        x = self.up32(x)  # (B,base,32,32)
        x = torch.cat([x, cond_pyr['s32']], dim=1)
        x = self.dec32(x)  # (B,base,32,32)

        # 64
        x = self.up64(x)  # (B,base,64,64)
        x = torch.cat([x, cond_pyr['s64']], dim=1)
        x = self.dec64(x)  # (B,base,64,64)

        # 70
        x = F.interpolate(x, size=(70, 70), mode='bilinear', align_corners=False)  # (B,base,70,70)
        x = self.dec70_in(x)
        x = torch.cat([x, cond_pyr['s70']], dim=1)
        x = self.dec70(x)  # (B,base,70,70)

        y = self.head(x)  # (B,1,70,70)

        if self.final_act == "tanh":
            y = torch.tanh(y)
        elif self.final_act == "sigmoid":
            y = torch.sigmoid(y)
        elif self.final_act == "softplus":
            y = F.softplus(y)
        # "none" -> 不做激活
        return y


if __name__ == "__main__":
    def test1():
        B = 2
        x = torch.randn(B, 1, 70, 70)
        y = torch.randn(B, 1, 1000, 70)

        conds = {'rms_vel': y, 'migrate': y, 'horizens': x, 'well_log': x, }

        model = CondFusionPyramid70()
        autoencoder = CondUnetAutoencoder()
        cond = model(conds)
        pred = autoencoder(x, cond)
        print(pred.shape)


    test1()
