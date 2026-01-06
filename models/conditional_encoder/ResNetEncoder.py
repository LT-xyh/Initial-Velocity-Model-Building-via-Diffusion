from typing import Tuple

import torch
from torch import nn

from diffusers.models.autoencoders.vae import Encoder
from utils.modules import Interpolation


class SimpleResNetCondEncoder(nn.Module):
    """
    简单 ResNet 编码器：
        input: 将四个条件输入resize到256后按通道维度拼接起来 [b, 4, 256, 256]
        output: 通道数128 shape为64 可以和 resize 后的 input 拼接起来输入到 U-net 中
    """

    def __init__(self,
                 in_channels=1,
                 mid_channels=64,  # 中间通道(初始层卷积)
                 out_channels=64,  # 输出的通道数
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D", ...),
                 block_out_channels: Tuple[int] = (64, ...),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 norm_num_groups: int = 32,
                 mid_block_add_attention: bool = True,
                 ):
        super().__init__()
        self.cond_key = ('migrate', 'rms_vel', 'well_log', 'horizens')
        self.init_convs = nn.ModuleDict({
            'migrate': nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            'rms_vel': nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            'well_log': nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            'horizens': nn.Conv2d(in_channels, mid_channels, 3, padding=1),
        })
        self.resize = Interpolation(shape=(256, 256))
        self.encoder = Encoder(
            in_channels=mid_channels * 4,  # 输入图像的通道数
            out_channels=out_channels,  # 编码器输出的潜在空间通道数
            down_block_types=down_block_types,  # 下采样块类型元组
            block_out_channels=block_out_channels,  # 每个模块输出的通道数
            layers_per_block=layers_per_block,  # 每个块中的层的数量
            act_fn=act_fn,  # 使用的激活函数
            norm_num_groups=norm_num_groups,  # 归一化层的组数
            double_z=False,  # 是否将输出通道数翻倍以用于VAE的均值和标准差
            mid_block_add_attention=mid_block_add_attention,  # 中间块是否包含注意力机制
        )

    def forward(self, input_dict: dict):
        cond_embedding = None
        for k in self.cond_key:
            cond = input_dict[k]  # [b, 1, w, h]
            cond = self.init_convs[k](cond)  # [b, 64, w, h]
            cond = self.resize(cond)  # [b, 64, 256, 256]
            if cond_embedding is None:
                cond_embedding = cond
            else:
                cond_embedding = torch.concat((cond, cond_embedding), dim=1)
        del input_dict
        x = self.encoder(cond_embedding)
        return x


if __name__ == "__main__":
    batch = 4
    device = 'cuda'
    input_dict = {
        'migrate': torch.randn(batch, 1, 1000, 70).to(device),  # 叠后偏移成像 (batch,1, 1000, 70)
        'rms_vel': torch.randn(batch, 1, 1000, 70).to(device),  # 时间域速度模型 (batch, 1000, 70)
        'model': torch.randn(batch, 1, 70, 70).to(device),  # 深度域速度模型 (batch, 70, 70)
        'well_log': torch.randn(batch, 1, 70, 70).to(device),  # 测井 (batch, 70, 70)
        'horizens': torch.randn(batch, 1, 70, 70).to(device)  # 层位 (batch, 70, 70)
    }

    # model = SimpleResNetCondEncoder(in_channels=1, out_channels=64).to(device)  # 0.57 m
    model = SimpleResNetCondEncoder(in_channels=1, mid_channels=64, out_channels=64,
                                    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
                                    block_out_channels=(256, 128, 64),
                                    ).to(device)
    print(f'参数量：{sum([p.numel() for p in model.parameters()]) / (2 ** 20):.2f} m')
    output = model(input_dict=input_dict)
    print("特征的形状:", output.shape)
