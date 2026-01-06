# cond_autoencoder_ae.py
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import Encoder, Decoder
from utils.modules import Interpolation, Reshape  # 你工程里的上采样模块


class CondAutoencoderAE(nn.Module):
    """
    条件自编码器（AE 版本，确定性）：
      - 潜空间侧拼接 latent_cond → 解码重建 x̂
      - 仅重建损失，无 KL
    """

    def __init__(
            self,
            input_shape: Tuple[int, ...] = (1, 70, 70),
            reshape: Tuple[int, ...] = (16, 64, 64),
            latent_cond_channels: int = 64,  # 潜空间条件通道（与 z 拼接）
            down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D", ...),
            down_block_out_channels: Tuple[int, ...] = (64, ...),
            up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D", ...),
            up_block_out_channels: Tuple[int, ...] = (64, ...),
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
        self.latent_cond_channels = latent_cond_channels

        # 1) 先把 (C=1,70,70) 变成与条件分辨率一致的 (C=reshape[0],64,64)
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], reshape[0], kernel_size=3, padding=1),
            Interpolation(reshape[1:]),  # -> (reshape[0], 64, 64)
        )
        # self.init_mlp = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=math.prod(input_shape), out_features=math.prod(reshape)),
        #     Reshape((-1, *reshape)),
        # )

        # 2) 编码器：注意 AE 不需要 double_z（不输出 μ、σ）
        self.encoder = Encoder(
            in_channels=reshape[0],
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=down_block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,  # ★ 与 VAE 的关键差异
            mid_block_add_attention=mid_block_add_attention,
        )

        # 3) 可选的潜空间投影（把 [z, latent_cond] 做个 1×1 对齐/混合）
        in_latent = latent_channels + latent_cond_channels
        self.latent_proj = nn.Conv2d(in_latent, in_latent, kernel_size=1)

        # 4) 解码器：直接吃拼接后的通道数
        self.decoder = Decoder(
            in_channels=in_latent,
            out_channels=reshape[0],
            up_block_types=up_block_types,
            block_out_channels=up_block_out_channels,
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
        # self.final_mlp = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=math.prod(reshape), out_features=math.prod(input_shape)),
        #     Reshape((-1, *input_shape)),
        # )

    @torch.no_grad()
    def _check_shapes(self, x, input_cond, latent_cond):
        # 仅用于早期排错（可以注释掉）
        assert x.shape[2:] == self.input_shape[1:], f"x HxW {x.shape[2:]} != {self.input_shape[1:]}"
        assert input_cond.shape[2:] == self.reshape[1:], f"input_cond HxW {input_cond.shape[2:]} != {self.reshape[1:]}"
        assert latent_cond.shape[2:] == (self.reshape[1] // 4, self.reshape[2] // 4) or True  # 视你的下采样深度

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x)  # (B, reshape[0], 64, 64)
        # h = self.init_mlp(x)
        z = self.encoder(h)  # (B, latent_channels, 16, 16) 取决于网络深度
        return z  # ★ 确定性 z（不再是分布）

    def decode(self, z: torch.Tensor, latent_cond: torch.Tensor) -> torch.Tensor:
        zc = torch.cat([z, latent_cond], dim=1)  # 潜空间侧条件
        zc = self.latent_proj(zc)
        y = self.decoder(zc)  # (B, reshape[0], 64, 64)
        y = self.final_conv(y)  # (B, C=1, 70, 70)
        # y = self.final_mlp(y)
        return y

    def forward(self, x, latent_cond, recon_weight=1.0, perceptual=None):
        """
        返回：recon, loss
        若传入 perceptual(x, recon) -> 标量，可自动叠加感知损失。
        """
        # self._check_shapes(x, latent_cond)
        z = self.encode(x)
        recon = self.decode(z, latent_cond)

        recon_loss = F.mse_loss(recon, x)
        loss = recon_weight * recon_loss

        if perceptual is not None:
            p_loss = perceptual(x, recon)
            loss = loss + p_loss

        return recon, loss


class CondAutoencoder(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, ...] = (1, 70, 70),
            reshape: Tuple[int, ...] = (16, 64, 64),
            in_cond_channels: int = 64,  # encoder的条件编码器通道
            latent_cond_channels: int = 64,  # latent中(diffusion, decoder)的条件编码器通道
            down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D", ...),
            up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D", ...),
            block_out_channels: Tuple[int, ...] = (64, ...),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 16,
            norm_num_groups: int = 32,
            sample_size: int = 32,
            scaling_factor: float = 0.18215,
            shift_factor: Optional[float] = None,
            latents_mean: Optional[Tuple[float]] = None,
            latents_std: Optional[Tuple[float]] = None,
            force_upcast: float = False,
            use_quant_conv: bool = True,
            use_post_quant_conv: bool = True,
            mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.autoencoder_kl = AutoencoderKL(
            in_channels=reshape[0] + in_cond_channels,
            out_channels=reshape[0],
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            force_upcast=False,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.init_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], reshape[0], kernel_size=3,
                      padding=1),
            Interpolation(reshape[1:]),
        )
        self.autoencoder_kl.encoder = Encoder(
            in_channels=reshape[0] + in_cond_channels,  # 输入图像的通道数
            out_channels=latent_channels,  # 编码器输出的潜在空间通道数
            down_block_types=down_block_types,  # 下采样块类型元组
            block_out_channels=block_out_channels,  # 每个模块输出的通道数
            layers_per_block=layers_per_block,  # 每个块中的层的数量
            act_fn=act_fn,  # 使用的激活函数
            norm_num_groups=norm_num_groups,  # 归一化层的组数
            # double_z=True,  # 是否将输出通道数翻倍以用于VAE的均值和标准差
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,  # 中间块是否包含注意力机制
        )
        self.autoencoder_kl.post_quant_conv = nn.Conv2d(latent_channels + latent_cond_channels,
                                                        latent_channels + latent_cond_channels, 1)
        # pass init params to Decoder
        self.autoencoder_kl.decoder = Decoder(
            in_channels=latent_channels + latent_cond_channels,
            out_channels=reshape[0],
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.final_conv = nn.Sequential(
            Interpolation(input_shape[1:]),
            nn.Conv2d(reshape[0], input_shape[0], kernel_size=3,
                      padding=1),
        )

    def encode(self, x, input_cond):
        x = self.init_conv(x)
        x = torch.cat([x, input_cond], dim=1)
        posterior = self.autoencoder_kl.encode(x).latent_dist
        return posterior

    def decode(self, latents, latent_cond):
        latents = torch.cat([latents, latent_cond], dim=1)
        reconstructions = self.autoencoder_kl.decode(latents, return_dict=True).sample
        reconstructions = self.final_conv(reconstructions)
        return reconstructions

    def forward(self, x, input_cond, latent_cond, kl_weight=1e-6):
        posterior = self.encode(x, input_cond)
        latents = posterior.sample()
        reconstructions = self.decode(latents, latent_cond)

        recon_loss = F.mse_loss(reconstructions, x, reduction="none")
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = recon_loss + kl_loss * kl_weight
        return reconstructions, loss,


def test_cond_autoencoder():
    # model = CondAutoencoder(
    #     input_shape=(1, 70, 70),
    #     reshape=(16, 64, 64),
    #     in_cond_channels=64,  # encoder的条件编码器通道
    #     latent_cond_channels=64,  # latent中(diffusion, decoder)的条件编码器通道
    #     down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
    #     up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    #     block_out_channels=(128, 256, 512),
    #     latent_channels=16,
    # )
    model = CondAutoencoderAE(
        input_shape=(1, 70, 70),
        reshape=(2, 64, 64),
        latent_cond_channels=64,  # latent中(diffusion, decoder)的条件编码器通道
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512),
        latent_channels=16,
    )
    input_tensor = torch.randn(4, 1, 70, 70)  # Batch of 1, 1 channel, 70x70
    latent_cond = torch.randn(4, 64, 16, 16)
    output, _ = model(input_tensor, latent_cond)
    assert output.shape == (4, 1, 70, 70), f"Expected output shape (4, 1, 70, 70), got {output.shape}"

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Test passed!")


if __name__ == "__main__":
    test_cond_autoencoder()
