import math

import torch
import torch.nn.functional as F
from lpips import lpips
from pytorch_msssim import SSIM
from torch import nn

from diffusers.training_utils import EMAModel
from lightning_modules.base_lightning import BaseLightningModule
from models.Autoencoder.Autoencoder import AutoencoderAE
from models.Autoencoder.AutoencoderKLInterpolation import AutoencoderKLInterpolation


class AutoencoderKLLightning(BaseLightningModule):
    def __init__(self, conf):
        if conf.datasets.use_normalize == '-1_1':
            data_range = 2.0
        else:
            data_range = 1.0
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr, data_range=data_range)
        self.conf = conf
        self.vae = AutoencoderKLInterpolation(
            latent_channels=conf.autoencoder_conf.latent_channels,
            depth_vel_shape=conf.datasets.depth_velocity.shape,
            depth_vel_reshape=conf.autoencoder_conf.reshape,
            down_block_types=conf.autoencoder_conf.down_block_types,
            up_block_types=conf.autoencoder_conf.up_block_types,
            block_out_channels=conf.autoencoder_conf.block_out_channels,
        )
        self.ema = None

        # === KL anneal config ===
        ka = getattr(conf.training, "kl_anneal", None)
        self.ka = {
            "strategy": (ka.strategy if ka else "linear_epoch"),
            "warmup_epochs": (ka.warmup_epochs if ka else max(1, int(conf.training.max_epochs * 0.2))),
            "start": (ka.start if ka else 0.0),
            "end": (ka.end if ka else conf.training.loss.kl_weight),
            "cycles": (ka.cycles if ka else 3),
            "ratio": (ka.ratio if ka else 0.5),
            "free_bits": (ka.free_bits if ka else 0.0),
        }
        self.register_buffer("kl_beta", torch.tensor(float(self.ka["start"])))  # 当前 β

    def setup(self, stage):
        super().setup(stage)
        if self.conf.training.use_ema:
            self.ema = EMAModel(parameters=self.parameters(),
                                use_ema_warmup=True, foreach=True,
                                power=0.75, device=self.device)

    # —— 每个 epoch 开头刷新一次 β（也可改成按 step 刷新）——
    def on_train_epoch_start(self):
        self.kl_beta.fill_(self._compute_beta_epoch(self.current_epoch))

    def _compute_beta_epoch(self, epoch: int) -> float:
        strat = self.ka["strategy"]
        start, end = float(self.ka["start"]), float(self.ka["end"])
        warm = int(self.ka["warmup_epochs"])
        if strat == "none":
            return end

        if strat == "linear_epoch":
            t = min(1.0, (epoch + 1) / max(1, warm))
            return start + t * (end - start)

        if strat == "cosine_epoch":
            if epoch + 1 <= warm:
                # 0 → 1 的半余弦升温
                t = 0.5 * (1 - math.cos(math.pi * (epoch + 1) / max(1, warm)))
            else:
                t = 1.0
            return start + t * (end - start)

        if strat == "cyclic_epoch":
            cycles = int(self.ka["cycles"])
            rise_ratio = float(self.ka["ratio"])
            total = max(1, self.trainer.max_epochs)
            cycle_len = max(1, total // max(1, cycles))
            pos_in_cycle = (epoch % cycle_len) + 1
            rise_len = max(1, int(cycle_len * rise_ratio))
            if pos_in_cycle <= rise_len:
                t = pos_in_cycle / rise_len
            else:
                t = 1.0
            return start + t * (end - start)

        return end  # fallback

    # —— 你原有的 training_step 里，把 KL 权重替换为 self.kl_beta.item() ——
    def training_step(self, batch, batch_idx):
        depth_velocity = batch.pop('depth_vel')

        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        reconstructions = self.vae.decode(latents)

        l1_loss = F.l1_loss(reconstructions, depth_velocity)
        mse_loss = F.mse_loss(reconstructions, depth_velocity)

        # KL 原始项
        kl_map = posterior.kl()  # 形状可能是 [B, C, H, W] 或 [B, ...]
        kl_loss_raw = kl_map.mean()

        # ——（可选）free-bits/min-rate：给 KL 一个最小速率阈值，避免过小——
        fb = float(self.ka["free_bits"])
        if fb > 0.0:
            B = kl_map.shape[0]
            kl_per_sample = kl_map.view(B, -1).sum(dim=1)  # 每个样本 KL 总量（nats）
            kl_loss = torch.clamp(kl_per_sample - fb, min=0.0).mean()
        else:
            kl_loss = kl_loss_raw

        beta = float(self.kl_beta.item())
        loss = (l1_loss * self.conf.training.loss.l1_weight
                + mse_loss * self.conf.training.loss.mse_weight
                + kl_loss * beta)

        self.log('train/loss', loss.detach().item(), on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=self.conf.training.dataloader.batch_size)
        self.log_dict({
            'train/MAE': l1_loss.detach().item(),
            'train/MSE': mse_loss.detach().item(),
            'train/KL_raw': kl_loss_raw.detach().item(),
            'train/KL_used': kl_loss.detach().item(),
            'train/beta': beta,
        }, on_step=True, on_epoch=False, prog_bar=False, batch_size=self.conf.training.dataloader.batch_size)

        if self.conf.training.use_ema:
            self.ema.step(self.vae.parameters())
        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self.vae.parameters());
            self.ema.copy_to(self.vae.parameters())

        depth_velocity = batch.pop('depth_vel')
        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        reconstructions = self.vae.decode(latents)

        l1_loss = F.l1_loss(reconstructions, depth_velocity)
        mse_loss = F.mse_loss(reconstructions, depth_velocity)
        kl_loss = posterior.kl().mean()

        beta = float(self.kl_beta.item())  # 验证集同一轮使用同一 β
        loss = (l1_loss * self.conf.training.loss.l1_weight
                + mse_loss * self.conf.training.loss.mse_weight
                + kl_loss * beta)

        self.log('val/loss', loss.detach().item(), on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=self.conf.training.dataloader.batch_size)
        self.log_dict({
            'val/MAE': l1_loss.detach().item(),
            'val/MSE': mse_loss.detach().item(),
            'val/KL': kl_loss.detach().item(),
            'val/beta': beta,
        }, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.conf.training.dataloader.batch_size)

        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity, reconstructions)

        if self.conf.training.use_ema:
            self.ema.restore(self.vae.parameters())
        return loss


class AutoencoderAELightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf
        self.ae = AutoencoderAE(
            input_shape=conf.datasets.depth_velocity.shape,
            reshape=conf.autoencoder_conf.reshape,
            down_block_types=conf.autoencoder_conf.down_block_types,
            up_block_types=conf.autoencoder_conf.up_block_types,
            block_out_channels=conf.autoencoder_conf.block_out_channels,
            latent_channels=conf.autoencoder_conf.latent_channels,
        )
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
        if self.conf.training.ssim_weight > 0:
            self.ssim_module = SSIM(data_range=1.0, size_average=True, win_size=11, win_sigma=1.5,
                                    channel=1, spatial_dims=2, K=(0.01, 0.03), nonnegative_ssim=True
                                    )
        self.ema = None

    def setup(self, stage):
        super().setup(stage)
        if self.conf.training.use_ema:
            self.ema = EMAModel(parameters=self.parameters(),
                                use_ema_warmup=True, foreach=True,
                                power=0.75, device=self.device)

    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch['model']
        del batch

        # 2. 模型
        latent_z = self.ae.encode(depth_velocity)
        reconstructions = self.ae.decode(latent_z)

        # 3. 损失
        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")  # 计算重建损失(MSE)
        with torch.no_grad():  # 不需要计算感知损失的梯度
            p_loss = self.perceptual_loss(reconstructions, depth_velocity)
        if self.conf.training.ssim_weight == 0:  # 如果没有启用SSIM损失，则将SSIM损失设置为0
            ssim_loss = 0
        else:
            ssim_loss = 1 - self.ssim_module(depth_velocity, reconstructions)
        # 组合总损失：重建损失 + KL损失(带权重) + 感知损失(带权重)
        loss = (recon_loss
                + p_loss * self.conf.training.perceptual_weight
                + ssim_loss * self.conf.training.ssim_weight).mean()
        self.log('train/loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True)
        logs = {
            'train/MSE': recon_loss.detach().mean().item(),
            'train/lpips': p_loss.detach().mean().item(),
        }
        if self.conf.training.ssim_weight > 0:
            logs.update({'train/ssim_loss': ssim_loss.detach().mean().item()})  # SSIM损失
        self.log_dict(logs)

        # 4. 评价指标
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity, reconstructions)

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self.ae.parameters())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self.ae.parameters())
            self.ema.copy_to(self.ae.parameters())

        # 1. 数据
        depth_velocity = batch['model']
        del batch

        # 2. 模型
        latent_z = self.ae.encode(depth_velocity)
        reconstructions = self.ae.decode(latent_z)

        # 3. 损失
        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")  # 计算重建损失(MSE)
        with torch.no_grad():  # 不需要计算感知损失的梯度
            p_loss = self.perceptual_loss(reconstructions, depth_velocity)
        if self.conf.training.ssim_weight == 0:  # 如果没有启用SSIM损失，则将SSIM损失设置为0
            ssim_loss = 0
        else:
            ssim_loss = 1 - self.ssim_module(depth_velocity, reconstructions)
        # 组合总损失：重建损失 + KL损失(带权重) + 感知损失(带权重)
        loss = (recon_loss
                + p_loss * self.conf.training.perceptual_weight
                + ssim_loss * self.conf.training.ssim_weight
                ).mean()
        logs = {
            'val/loss': loss.detach().item(),
            'val/MSE': recon_loss.detach().mean().item(),
            'val/lpips': p_loss.detach().mean().item(),
        }
        if self.conf.training.ssim_weight > 0:
            logs.update({'val/ssim_loss': ssim_loss.detach().mean().item()})  # SSIM损失
        self.log_dict(logs)

        # 4. 评价指标
        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity, reconstructions)

        if self.conf.training.use_ema:
            self.ema.restore(self.ae.parameters())
        return loss


class TestLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(conf.datasets.depth_velocity.shape), math.prod(conf.autoencoder_conf.reshape)),
            # Reshape(conf.autoencoder_conf.reshape),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(conf.autoencoder_conf.reshape), math.prod(conf.datasets.depth_velocity.shape)),
            # Reshape(conf.datasets.depth_velocity.shape),
        )

        self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
        # self.ema = None

    def setup(self, stage):
        super().setup(stage)
        # if self.conf.training.use_ema:
        #     self.ema = EMAModel(parameters=self.parameters(),
        #                         use_ema_warmup=True, foreach=True,
        #                         power=0.75, device=self.device)

    def training_step(self, batch, batch_idx):
        depth_velocity = batch['model']
        del batch

        latents = self.encoder(depth_velocity)
        reconstructions = self.decoder(latents)

        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")  # 计算重建损失(MSE)
        with torch.no_grad():  # 不需要计算感知损失的梯度
            p_loss = self.perceptual_loss(reconstructions, depth_velocity)
        # 组合总损失：重建损失 + KL损失(带权重) + 感知损失(带权重)
        loss = (recon_loss
                + p_loss * self.conf.training.perceptual_weight).mean()
        self.log('train/loss', loss.detach().item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log_dict({
            'train/MSE': recon_loss.detach().mean().item(),  # 均方误差
            'train/lpips': p_loss.detach().mean().item(),  # 感知损失
        })

        depth_velocity = self.normalize_to_one_to_neg_one(depth_velocity)
        reconstructions = self.normalize_to_one_to_neg_one(reconstructions)
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity, reconstructions)

        # if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
        #     self.ema.step(self.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e5)
        self.log("train/grad_norm", grad_norm)
        return loss

    def validation_step(self, batch, batch_idx):
        # if self.conf.training.use_ema:
        #     self.ema.store(self.parameters())
        #     self.ema.copy_to(self.parameters())

        depth_velocity = batch['model']
        del batch

        latents = self.encoder(depth_velocity)
        reconstructions = self.decoder(latents)

        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")
        with torch.no_grad():
            p_loss = self.perceptual_loss(reconstructions, depth_velocity)

        loss = (recon_loss
                + p_loss * self.conf.training.perceptual_weight).mean()

        self.log_dict({
            'val/loss': loss.detach().item(),
            'val/MSE': recon_loss.detach().mean().item(),
            'val/lpips': p_loss.detach().mean().item(),
        }, on_step=False, on_epoch=True, prog_bar=True)

        depth_velocity = self.normalize_to_one_to_neg_one(depth_velocity)
        reconstructions = self.normalize_to_one_to_neg_one(reconstructions)
        # 记录验证指标
        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity, reconstructions)

        # if self.conf.training.use_ema:
        #     self.ema.restore(self.parameters())
        return loss
