import torch
import torch.nn.functional as F
from diffusers import EMAModel
from lpips import lpips
from pytorch_msssim import SSIM

from lightning_modules.base_lightning import BaseLightningModule
from models.Autoencoder.CondUnetAutoencoder import CondUnetAutoencoder
from models.conditional_encoder.CondFusionPyramid70 import CondFusionPyramid70


class CondUnetAutoencoderLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf
        self.ae = CondUnetAutoencoder()
        # 条件编码器
        self.cond_encoder = CondFusionPyramid70()
        self.use_lpips = self.conf.training.perceptual_weight > 0
        if self.use_lpips:
            self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
        self.use_ssim = self.conf.training.ssim_weight > 0
        if self.use_ssim:
            self.ssim_module = SSIM(data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, channel=1,
                                    spatial_dims=2, K=(0.01, 0.03), nonnegative_ssim=True)
        self.ema = None

    def setup(self, stage):
        super().setup(stage)
        if self.conf.training.use_ema:
            self._ema_parameters = [p for p in self.parameters() if p.requires_grad]
            if self.ema is None:
                self.ema = EMAModel(parameters=self._ema_parameters, use_ema_warmup=True, foreach=True, power=0.75,
                                    device=self.device)

    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('model')
        depth_velocity = self.normalize_to_neg_one_to_one(depth_velocity)

        cond_embedding = self.cond_encoder(batch)
        del batch

        # 2. 模型
        latent_z = self.ae.encode(depth_velocity)
        reconstructions = self.ae.decode(latent_z, cond_embedding)

        # 3. 损失
        depth_velocity = self.unnormalize_from_neg_one_to_one(depth_velocity)
        reconstructions = self.unnormalize_from_neg_one_to_one(reconstructions)
        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")  # 计算重建损失(MSE)
        if self.use_lpips:
            with torch.no_grad():  # 不需要计算感知损失的梯度
                p_loss = self.perceptual_loss(reconstructions, depth_velocity)
        else:
            p_loss = 0
        if self.use_ssim:  # 如果没有启用SSIM损失，则将SSIM损失设置为0
            ssim_loss = 1 - self.ssim_module(depth_velocity, reconstructions)
        else:
            ssim_loss = 0
        # 组合总损失：重建损失 + KL损失(带权重) + 感知损失(带权重)
        loss = (
                recon_loss + p_loss * self.conf.training.perceptual_weight + ssim_loss * self.conf.training.ssim_weight).mean()
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        logs = {'train/MSE': recon_loss.detach().mean(), }
        if self.use_lpips:
            logs.update({'train/lpips_loss': p_loss.detach().mean()})
        if self.use_ssim:
            logs.update({'train/ssim_loss': ssim_loss.detach().mean()})  # SSIM损失
        self.log_dict(logs)

        # 4. 评价指标
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self._ema_params())
            self.ema.copy_to(self._ema_params())

        # 1. 数据
        depth_velocity = batch.pop('model')
        depth_velocity = self.normalize_to_neg_one_to_one(depth_velocity)

        cond_embedding = self.cond_encoder(batch)
        del batch

        # 2. 模型
        latent_z = self.ae.encode(depth_velocity)
        reconstructions = self.ae.decode(latent_z, cond_embedding)

        # 3. 损失
        depth_velocity = self.unnormalize_from_neg_one_to_one(depth_velocity)
        reconstructions = self.unnormalize_from_neg_one_to_one(reconstructions)
        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")

        if self.use_lpips:
            with torch.no_grad():  # 不需要计算感知损失的梯度
                p_loss = self.perceptual_loss(reconstructions, depth_velocity)
        else:
            p_loss = 0

        if self.use_ssim:  # 如果没有启用SSIM损失，则将SSIM损失设置为0
            ssim_loss = 1 - self.ssim_module(depth_velocity, reconstructions)
        else:
            ssim_loss = 0
        loss = (
                recon_loss + p_loss * self.conf.training.perceptual_weight + ssim_loss * self.conf.training.ssim_weight).mean()

        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        logs = {'val/MSE': recon_loss.detach().mean(), }
        if self.use_lpips:
            logs.update({'val/lpips_loss': p_loss.detach().mean()})
        if self.use_ssim:
            logs.update({'val/ssim_loss': ssim_loss.detach().mean()})  # SSIM损失
        self.log_dict(logs)
        # 记录验证指标
        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss


class CondUnetAutoencoderZLightning(CondUnetAutoencoderLightning):
    def __init__(self, conf):
        super().__init__(conf)
        self.max_latent_recon_weight = conf.training.max_latent_recon_weight
        self.latent_recon_weight = 1e-4

    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('model')
        depth_velocity = self.normalize_to_neg_one_to_one(depth_velocity)

        cond_embedding = self.cond_encoder(batch)
        del batch

        # 2. 模型
        latent_z = self.ae.encode(depth_velocity)
        recon_latent_z = self.cond_encoder.decoder(cond_embedding['s16'])
        reconstructions = self.ae.decode(latent_z, cond_embedding)

        # 3. 损失
        depth_velocity = self.unnormalize_from_neg_one_to_one(depth_velocity)
        reconstructions = self.unnormalize_from_neg_one_to_one(reconstructions)
        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")  # 计算重建损失(MSE)

        latent_recon_loss = F.mse_loss(recon_latent_z, latent_z)  # 计算Z向量重建损失(MSE)

        if self.use_lpips:
            with torch.no_grad():  # 不需要计算感知损失的梯度
                p_loss = self.perceptual_loss(reconstructions, depth_velocity)
        else:
            p_loss = 0
        if self.use_ssim:  # 如果没有启用SSIM损失，则将SSIM损失设置为0
            ssim_loss = 1 - self.ssim_module(depth_velocity, reconstructions)
        else:
            ssim_loss = 0
        # 组合总损失：重建损失 + KL损失(带权重) + 感知损失(带权重)
        self.latent_recon_weight = min(self.max_latent_recon_weight,
                                       self.latent_recon_weight + self.current_epoch * 0.015)
        loss = (
                recon_loss + p_loss * self.conf.training.perceptual_weight + ssim_loss * self.conf.training.ssim_weight + latent_recon_loss * self.latent_recon_weight).mean()
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        logs = {'train/MSE': recon_loss.detach().mean(),
                'train/z_MSE': latent_recon_loss.detach().mean(), }
        if self.use_lpips:
            logs.update({'train/lpips_loss': p_loss.detach().mean()})
        if self.use_ssim:
            logs.update({'train/ssim_loss': ssim_loss.detach().mean()})  # SSIM损失
        self.log_dict(logs)

        # 4. 评价指标
        self.train_metrics.update(depth_velocity, reconstructions)
        self._last_train_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self._ema_params())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self._ema_params())
            self.ema.copy_to(self._ema_params())

        # 1. 数据
        depth_velocity = batch.pop('model')
        depth_velocity = self.normalize_to_neg_one_to_one(depth_velocity)

        cond_embedding = self.cond_encoder(batch)
        del batch

        # 2. 模型
        latent_z = self.ae.encode(depth_velocity)
        recon_latent_z = self.cond_encoder.decoder(cond_embedding['s16'])
        reconstructions = self.ae.decode(latent_z, cond_embedding)

        # 3. 损失
        depth_velocity = self.unnormalize_from_neg_one_to_one(depth_velocity)
        reconstructions = self.unnormalize_from_neg_one_to_one(reconstructions)
        recon_loss = F.mse_loss(reconstructions, depth_velocity, reduction="none")  # 计算重建损失(MSE)

        latent_recon_loss = F.mse_loss(recon_latent_z, latent_z)  # 计算Z向量重建损失(MSE)

        if self.use_lpips:
            with torch.no_grad():  # 不需要计算感知损失的梯度
                p_loss = self.perceptual_loss(reconstructions, depth_velocity)
        else:
            p_loss = 0

        if self.use_ssim:  # 如果没有启用SSIM损失，则将SSIM损失设置为0
            ssim_loss = 1 - self.ssim_module(depth_velocity, reconstructions)
        else:
            ssim_loss = 0
        # 组合总损失：重建损失 + KL损失(带权重) + 感知损失(带权重)
        self.latent_recon_weight = min(self.max_latent_recon_weight,
                                       self.latent_recon_weight + self.current_epoch * 0.015)
        loss = (
                recon_loss + p_loss * self.conf.training.perceptual_weight + ssim_loss * self.conf.training.ssim_weight + latent_recon_loss * self.latent_recon_weight).mean()
        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        logs = {'val/MSE': recon_loss.detach().mean(), 'val/z_MSE': latent_recon_loss.detach().mean(), }
        if self.use_lpips:
            logs.update({'val/lpips_loss': p_loss.detach().mean()})
        if self.use_ssim:
            logs.update({'val/ssim_loss': ssim_loss.detach().mean()})  # SSIM损失
        self.log_dict(logs)
        # 记录验证指标
        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss
