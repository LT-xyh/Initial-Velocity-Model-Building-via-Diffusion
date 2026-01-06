from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# utils: center crop / pad
# -------------------------
def center_crop(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    _, _, H, W = x.shape
    oh, ow = out_hw
    if H == oh and W == ow:
        return x
    top = max((H - oh) // 2, 0)
    left = max((W - ow) // 2, 0)
    x = x[:, :, top:top + oh, left:left + ow]
    return x


def center_pad(x: torch.Tensor, out_hw: Tuple[int, int], value: float = 0.0) -> torch.Tensor:
    _, _, H, W = x.shape
    oh, ow = out_hw
    pad_h = max(oh - H, 0)
    pad_w = max(ow - W, 0)
    if pad_h == 0 and pad_w == 0:
        return x
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode="constant", value=value)


def pad_time_to(x: torch.Tensor, target_H: int, value: float = 0.0) -> torch.Tensor:
    # x: (B,C,H,W) pad on H
    _, _, H, _ = x.shape
    if H >= target_H:
        return x
    pad = target_H - H
    pad_top = pad // 2
    pad_bot = pad - pad_top
    return F.pad(x, (0, 0, pad_top, pad_bot), mode="constant", value=value)


# -------------------------
# basic blocks (BN + LeakyReLU)
# -------------------------
class ConvBNLReLU(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                                 nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True), )

    def forward(self, x): return self.net(x)


class DeconvBNLReLU(nn.Module):
    def __init__(self, cin, cout, k, s, p, out_pad=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=k, stride=s, padding=p, output_padding=out_pad, bias=False),
            nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True), )

    def forward(self, x): return self.net(x)


# -------------------------
# Generator (VelocityGAN-style)
# -------------------------
class VelocityGAN_Generator_MC(nn.Module):
    """
    Multi-constraint generator that keeps VelocityGAN's core design:
    - k×1 convs with stride(2×1) to reduce time dimension
    - 3×3 conv encoder to 8×8 then 8×8 conv -> 1×1
    - decoder via transposed conv blocks + center crop/pad + tanh
    Input:
      migrated_image: (B,1,1000,70)
      rms_vel      : (B,1,1000,70)
      horizon      : (B,1,70,70)
      well_log     : (B,1,70,70)   missing=-1 (after norm)
    Output:
      depth_vel_pred: (B,1,70,70) in [-1,1]
    """

    def __init__(self, in_ch: int = 5, base: int = 32, target_hw=(70, 70)):
        super().__init__()
        self.target_hw = target_hw

        # ---- Time-direction reduction (k×1, stride(2×1)) ----
        # Make time H from 1024 -> 64 in 4 downsamples.
        self.t1 = ConvBNLReLU(in_ch, base, k=(7, 1), s=(2, 1), p=(3, 0))  # 1024->512
        self.t2 = ConvBNLReLU(base, base * 2, k=(3, 1), s=(2, 1), p=(1, 0))  # 512->256
        self.t3 = ConvBNLReLU(base * 2, base * 4, k=(3, 1), s=(2, 1), p=(1, 0))  # 256->128
        self.t4 = ConvBNLReLU(base * 4, base * 8, k=(3, 1), s=(2, 1), p=(1, 0))  # 128->64

        # ---- 3×3 encoder (64×64 -> 32 -> 16 -> 8) ----
        def enc_block(cin, cout):
            return nn.Sequential(ConvBNLReLU(cin, cout, k=3, s=2, p=1), ConvBNLReLU(cout, cout, k=3, s=1, p=1), )

        self.e1 = enc_block(base * 8, base * 8)  # 64->32
        self.e2 = enc_block(base * 8, base * 8)  # 32->16
        self.e3 = enc_block(base * 8, base * 8)  # 16->8

        # ---- "eliminate spatial info" conv: 8×8 -> 1×1 ----
        self.conv_global = nn.Sequential(nn.Conv2d(base * 8, base * 16, kernel_size=8, stride=1, padding=0, bias=False),
                                         nn.BatchNorm2d(base * 16), nn.LeakyReLU(0.2, inplace=True), )

        # ---- Decoder (mirror idea): 1×1 -> 8 -> 16 -> 32 -> 64 ----
        self.d0 = DeconvBNLReLU(base * 16, base * 8, k=8, s=1, p=0)  # 1->8
        self.d1 = DeconvBNLReLU(base * 8, base * 4, k=4, s=2, p=1)  # 8->16
        self.d2 = DeconvBNLReLU(base * 4, base * 2, k=4, s=2, p=1)  # 16->32
        self.d3 = DeconvBNLReLU(base * 2, base, k=4, s=2, p=1)  # 32->64

        self.out_conv = nn.Conv2d(base, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, migrated_image: torch.Tensor, rms_vel: torch.Tensor, horizon: torch.Tensor,
                well_log: torch.Tensor) -> torch.Tensor:
        B = migrated_image.size(0)

        # ----- build conditional channels (keep single-stream generator) -----
        # well mask & neutralized well values
        eps = 1e-6

        # upsample depth constraints to time-grid (1000,70)
        horizon_up = F.interpolate(horizon, size=(1000, 70), mode="bilinear")
        well_log_up = F.interpolate(well_log, size=(1000, 70), mode="bilinear", align_corners=False)

        x = torch.cat([migrated_image, rms_vel, horizon_up, well_log_up], dim=1)  # (B,5,1000,70)

        # pad time to 1024 to match downsampling to 64 exactly
        x = pad_time_to(x, 1024, value=0.0)  # (B,5,1024,70)

        # width crop 70 -> 64 to form 64×64 square (paper later uses center crop too)
        # first reduce time height to 64
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)  # (B,C,64,70)
        x = center_crop(x, (64, 64))  # (B,C,64,64)

        # 3×3 encoder to 8×8 then global conv to 1×1
        x = self.e1(x)  # 32×32
        x = self.e2(x)  # 16×16
        x = self.e3(x)  # 8×8
        x = self.conv_global(x)  # 1×1

        # decode back to 64×64
        x = self.d0(x)  # 8×8
        x = self.d1(x)  # 16×16
        x = self.d2(x)  # 32×32
        x = self.d3(x)  # 64×64
        x = self.out_conv(x)  # 64×64

        # pad/crop to final 70×70 (consistent with "center crop to desired dimension")
        x = center_pad(x, self.target_hw, value=0.0)  # 70×70
        x = center_crop(x, self.target_hw)  # safety
        return self.tanh(x)


# -------------------------
# Discriminator (PatchGAN-like, patch grid ~ 4×4)
# -------------------------
class VelocityGAN_Discriminator_Patch4(nn.Module):
    """
    PatchGAN-style critic:
    - 4 times MaxPool2d(2) => 70→35→17→8→4 (approx patch grid 4×4)
    - output a (B,1,4,4) score map, then we take mean as scalar critic value
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()

        def block(cin, cout, pool=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(cout),
                      nn.LeakyReLU(0.2, inplace=True), ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.b1 = block(in_ch, base, pool=True)  # 70->35
        self.b2 = block(base, base * 2, pool=True)  # 35->17
        self.b3 = block(base * 2, base * 4, pool=True)  # 17->8
        self.b4 = block(base * 4, base * 8, pool=True)  # 8->4
        self.b5 = block(base * 8, base * 8, pool=False)  # keep 4->4
        self.head = nn.Conv2d(base * 8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        x = self.b1(v)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.head(x)  # (B,1,4,4)

    def score(self, v: torch.Tensor) -> torch.Tensor:
        # scalar per sample for WGAN
        p = self.forward(v)
        return p.mean(dim=(1, 2, 3))  # (B,)


# -------------------------
# WGAN-GP losses (as in paper)
# -------------------------
def wgan_gp_discriminator_loss(D: VelocityGAN_Discriminator_Patch4, real: torch.Tensor, fake: torch.Tensor,
                               gp_lambda: float = 10.0) -> torch.Tensor:
    # Wasserstein loss + gradient penalty
    B = real.size(0)
    real_score = D.score(real)
    fake_score = D.score(fake.detach())
    loss_w = (fake_score - real_score).mean()

    # gradient penalty
    eps = torch.rand(B, 1, 1, 1, device=real.device, dtype=real.dtype)
    x_hat = eps * real + (1 - eps) * fake.detach()
    x_hat.requires_grad_(True)
    hat_score = D.score(x_hat).sum()  # sum to get scalar for autograd
    grads = torch.autograd.grad(hat_score, x_hat, create_graph=True)[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return loss_w + gp_lambda * gp


def generator_loss(D: VelocityGAN_Discriminator_Patch4, fake: torch.Tensor, real: torch.Tensor, l1_w: float = 50.0,
                   l2_w: float = 100.0) -> torch.Tensor:
    adv = -D.score(fake).mean()
    l1 = F.l1_loss(fake, real)
    l2 = F.mse_loss(fake, real)
    return adv + l1_w * l1 + l2_w * l2


# -------------------------
# quick self-test
# -------------------------
import torch


def smoke_test_velocitygan_mc(device=None):
    # 0) device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)

    # 1) shapes
    B = 2
    T, X = 1000, 70
    Z = 70

    # 2) build inputs in [-1, 1]
    migrated_image = torch.randn(B, 1, T, X, device=device).clamp(-1, 1)
    rms_vel = torch.randn(B, 1, T, X, device=device).clamp(-1, 1)

    # horizon: binary in {-1, +1}, where +1 indicates interface
    # (这里只是随机生成一些界面像素用于测试)
    horizon = -torch.ones(B, 1, Z, X, device=device)
    rand_mask = (torch.rand(B, 1, Z, X, device=device) < 0.05)  # 5% pixels are horizons
    horizon[rand_mask] = 1.0

    # well_log: missing = -1 (after your normalization), well positions = random in [-1,1]
    well_log = -torch.ones(B, 1, Z, X, device=device)
    # create a few well columns (e.g., x = 10, 35, 60)
    well_cols = [10, 35, 60]
    for xc in well_cols:
        well_log[:, :, :, xc] = torch.randn(B, 1, Z, device=device).clamp(-1, 1)

    # ground-truth depth velocity (just random target for smoke test)
    depth_vel_gt = torch.randn(B, 1, Z, X, device=device).clamp(-1, 1)

    # 3) build models
    G = VelocityGAN_Generator_MC(in_ch=5, base=32, target_hw=(70, 70)).to(device).train()
    D = VelocityGAN_Discriminator_Patch4(in_ch=1, base=32).to(device).train()

    # 4) optimizers
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # -------------------------
    # Step A: update D once
    # -------------------------
    with torch.no_grad():
        fake = G(migrated_image, rms_vel, horizon, well_log)  # (B,1,70,70)

    lossD = wgan_gp_discriminator_loss(D, real=depth_vel_gt, fake=fake, gp_lambda=10.0)

    optD.zero_grad(set_to_none=True)
    lossD.backward()
    optD.step()

    # -------------------------
    # Step B: update G once
    # -------------------------
    fake = G(migrated_image, rms_vel, horizon, well_log)  # re-generate with grad
    lossG = generator_loss(D, fake=fake, real=depth_vel_gt, l1_w=50.0, l2_w=100.0)

    optG.zero_grad(set_to_none=True)
    lossG.backward()
    optG.step()

    # 5) print sanity info
    with torch.no_grad():
        d_real = D.score(depth_vel_gt).mean().item()
        d_fake = D.score(fake).mean().item()

    print(f"[OK] device={device}")
    print(f"fake shape: {tuple(fake.shape)} (expect (B,1,70,70))")
    print(f"D(real) mean: {d_real:.4f}, D(fake) mean: {d_fake:.4f}")
    print(f"lossD: {lossD.item():.4f}, lossG: {lossG.item():.4f}")


if __name__ == "__main__":
    smoke_test_velocitygan_mc()
