import torch
import torch.nn.functional as F
from diffusers import EMAModel
from lpips import lpips
from pytorch_msssim import SSIM
from torch import nn

from lightning_modules.base_lightning import BaseLightningModule
from models.Autoencoder.MyAE import MyConditionalAE
from models.Autoencoder.cond_autoencder import CondAutoencoderAE
from models.conditional_encoder.ConditionalFusion import MultiModalFusion
from models.conditional_encoder.HorizonEncoder import HorizonEncoder
from models.conditional_encoder.MigratedImagingEncoder import MigratedImagingEncoder
from models.conditional_encoder.RMSEncoder import RMSVelocityEncoder
from models.conditional_encoder.WellLogEncoder import WellLogEncoder


class CondAutoencoderLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf
        self.ae = CondAutoencoderAE(input_shape=conf.datasets.depth_velocity.shape,
                                    reshape=conf.autoencoder_conf.reshape,
                                    latent_cond_channels=conf.latent_cond_encoder.out_channels,
                                    # latent中(diffusion, decoder)的条件编码器通道
                                    down_block_types=conf.autoencoder_conf.down_block_types,
                                    down_block_out_channels=conf.autoencoder_conf.down_block_out_channels,
                                    up_block_types=conf.autoencoder_conf.up_block_types,
                                    up_block_out_channels=conf.autoencoder_conf.up_block_out_channels,
                                    latent_channels=conf.autoencoder_conf.latent_channels, )
        # 条件编码器
        self.cond_encoder = nn.ModuleDict(
            {'rms_encoder': RMSVelocityEncoder(), 'migrated_imaging_encoder': MigratedImagingEncoder(),
             'horizon_encoder': HorizonEncoder(), 'well_log_encoder': WellLogEncoder(), })
        # 条件融合
        self.multi_fusion = MultiModalFusion(in_channels={'rms': 32, 'migrated': 64, 'horizon': 16, 'well': 16},
                                             C_out=64,  # 统一条件嵌入通道，推荐 64
                                             C_mid=128, return_vector_dim=0,  # 如需 AdaIN/FiLM 的全局向量；不需要可设 0
                                             modality_dropout_p=0  # 训练期可开；推理期会自动关闭
                                             )
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
        depth_velocity = batch['model']
        depth_velocity = self.normalize_to_neg_one_to_one(depth_velocity)

        cond_embedding = {'rms': self.cond_encoder['rms_encoder'](batch['rms_vel']),
                          'migrated': self.cond_encoder['migrated_imaging_encoder'](batch['migrate']),
                          'horizon': self.cond_encoder['horizon_encoder'](batch['horizens']),
                          'well': self.cond_encoder['well_log_encoder'](batch['well_log'])}
        del batch

        # 2. 模型
        latent_cond = self.multi_fusion(cond_embedding)['map']

        latent_z = self.ae.encode(depth_velocity)
        reconstructions = self.ae.decode(latent_z, latent_cond)

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
        depth_velocity = batch['model']
        depth_velocity = self.normalize_to_neg_one_to_one(depth_velocity)

        cond_embedding = {'rms': self.cond_encoder['rms_encoder'](batch['rms_vel']),
                          'migrated': self.cond_encoder['migrated_imaging_encoder'](batch['migrate']),
                          'horizon': self.cond_encoder['horizon_encoder'](batch['horizens']),
                          'well': self.cond_encoder['well_log_encoder'](batch['well_log'])}
        del batch

        # 2. 模型
        latent_cond = self.multi_fusion(cond_embedding)['map']

        latent_z = self.ae.encode(depth_velocity)
        reconstructions = self.ae.decode(latent_z, latent_cond)

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

        self.log('val/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        logs = {'val/MSE': recon_loss.detach().mean(), }
        if self.use_lpips:
            logs.update({'val/lpips_loss': p_loss.detach().mean()})
        if self.use_ssim:
            logs.update({'val/ssim_loss': ssim_loss.detach().mean()})  # SSIM损失
        self.log_dict(logs)
        # depth_velocity = self.normalize_to_one_to_neg_one(depth_velocity)
        # reconstructions = self.normalize_to_one_to_neg_one(reconstructions)
        # 记录验证指标
        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss


class MyConditionalAELightning(CondAutoencoderLightning):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.ae = MyConditionalAE(in_channels=self.conf.datasets.depth_velocity.shape[0],
                                  latent_channels=self.conf.autoencoder_conf.latent_channels,
                                  in_size=self.conf.datasets.depth_velocity.shape[1:], base_channel=64,
                                  cond_channels=self.conf.latent_cond_encoder.out_channels, )
