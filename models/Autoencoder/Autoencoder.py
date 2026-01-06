from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.autoencoders.vae import Encoder, Decoder
from models.Autoencoder.AutoencoderKLInterpolation import AutoencoderKLInterpolation
from utils.modules import Interpolation


class AutoencoderAE(nn.Module):
    """
    自编码器（AE 版本，确定性）：
      - 仅重建损失，无 KL
    """

    def __init__(
            self,
            input_shape: Tuple[int, ...] = (1, 70, 70),
            reshape: Tuple[int, ...] = (16, 64, 64),
            down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D", ...),
            up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D", ...),
            block_out_channels: Tuple[int, ...] = (64, ...),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 16,
            norm_num_groups: int = 32,
            mid_block_add_attention: bool = True,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.reshape = reshape
        self.latent_channels = latent_channels

        # 1) 先把 (C=1,70,70) 变成与条件分辨率一致的 (C=reshape[0],64,64)
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], reshape[0], kernel_size=3, padding=1),
            Interpolation(reshape[1:]),  # -> (reshape[0], 64, 64)
        )

        # 2) 编码器：注意 AE 不需要 double_z（不输出 μ、σ）
        self.encoder = Encoder(
            in_channels=reshape[0],
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,  # ★ 与 VAE 的关键差异
            mid_block_add_attention=mid_block_add_attention,
        )

        # 4) 解码器：直接吃拼接后的通道数
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=reshape[0],
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        # 5) 还原到输入分辨率与通道
        self.final_conv = nn.Sequential(
            Interpolation(input_shape[1:]),  # -> (reshape[0], 70, 70)
            nn.Conv2d(reshape[0], input_shape[0], kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_feat = self.init_conv(x)  # (B, reshape[0], 64, 64)
        z = self.encoder(x_feat)  # (B, latent_channels, 16, 16) 取决于网络深度
        return z  # ★ 确定性 z（不再是分布）

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z)  # (B, reshape[0], 64, 64)
        y = self.final_conv(y)  # (B, C=1, 70, 70)
        return y

    def forward(self, x, recon_weight=1.0, perceptual=None):
        """
        返回：recon, loss
        若传入 perceptual(x, recon) -> 标量，可自动叠加感知损失。
        """
        # self._check_shapes(x, input_cond, latent_cond)
        z = self.encode(x)
        recon = self.decode(z)

        recon_loss = F.mse_loss(recon, x)
        loss = recon_weight * recon_loss

        if perceptual is not None:
            p_loss = perceptual(x, recon)
            loss = loss + p_loss

        return recon, loss


class AutoencoderMLP(AutoencoderKLInterpolation):
    def __init__(self, conf):
        super().__init__(conf)

    def forward(self, x):
        posterior = self.encode(x)
        latents = posterior.mode()  # 使用对角高斯分布的均值 (均值)
        reconstructions = self.decode(latents)

        recon_loss = F.mse_loss(reconstructions, x, reduction="none")
        loss = recon_loss
        return reconstructions, loss,


def test_cond_autoencoder():
    model = AutoencoderAE(
        input_shape=(1, 70, 70),
        reshape=(16, 64, 64),

        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512),
        latent_channels=16,
    )
    input_tensor = torch.randn(4, 1, 70, 70)  # Batch of 1, 1 channel, 70x70
    output, _ = model(input_tensor)
    assert output.shape == (4, 1, 70, 70), f"Expected output shape (4, 1, 70, 70), got {output.shape}"

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Test passed!")

if __name__ == "__main__":
    test_cond_autoencoder()