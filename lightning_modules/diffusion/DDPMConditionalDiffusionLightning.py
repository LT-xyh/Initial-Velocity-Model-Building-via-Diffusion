import torch.nn.functional as F

from diffusers import EMAModel
from lightning_modules.Autoencoder.autoencoder_kl_lightning import AutoencoderKLLightning
from lightning_modules.base_lightning import BaseLightningModule
from models.conditional_encoder.CondFusionPyramid70 import CondFusionPyramid70
from models.diffusion.DiffusionConditionedUNet import LatentConditionalDiffusion


class DDPMConditionalDiffusionLightning(BaseLightningModule):
    """
    DDPM conditional latent diffusion:
        数据归一化至[-1, 1]

    """

    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf
        # 1. 加载预训练的autoencoder
        autoencoder = AutoencoderKLLightning.load_from_checkpoint(conf.autoencoder_conf.autoencoder_checkpoint_path)
        for param in autoencoder.parameters():
            param.requires_grad = False
        self.vae = autoencoder.vae
        del autoencoder

        # 2. latent diffusion
        self.ldm = LatentConditionalDiffusion(scheduler_type=self.conf.latent_diffusion.scheduler.scheduler_type,
                                              num_train_timesteps=self.conf.latent_diffusion.scheduler.num_train_timesteps)

        # 3. Conditional encoder
        self.ldm_cond_encoder = CondFusionPyramid70()

    def setup(self, stage):
        super().setup(stage)
        if self.conf.training.use_ema:
            self.ema = EMAModel(parameters=self.parameters(),
                                use_ema_warmup=True, foreach=True,
                                power=0.75, device=self.device)

    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')

        ldm_cond_embedding = self.ldm_cond_encoder(batch)['s16']
        del batch

        # 2. 模型
        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        ldm_dict = self.ldm.training_loss(x0=latents, cond=ldm_cond_embedding, loss_type="mse")
        # recon_z = ldm_dict['x0_pred']
        # reconstructions = self.vae.decode(recon_z)

        # 3. 损失
        loss = ldm_dict['loss']
        self.log('train/loss', loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True)

        if self.conf.training.use_ema:  # 如果启用了EMA，则更新EMA参数
            self.ema.step(self.parameters())

        return loss

    def validation_step(self, batch, batch_idx):
        if self.conf.training.use_ema:
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())

        # 1. 数据
        depth_velocity = batch.pop('depth_vel')

        ldm_cond_embedding = self.ldm_cond_encoder(batch)['s16']
        del batch

        # 2. 模型
        recon_z = self.ldm.sample(cond=ldm_cond_embedding, x_size=(depth_velocity.shape[0], 16, 16, 16))
        reconstructions = self.vae.decode(recon_z)

        # 3. 损失
        mse = F.mse_loss(depth_velocity, reconstructions)
        mae = F.l1_loss(depth_velocity, reconstructions)
        self.log('val/mse', mse.detach().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mae', mae.detach().item(), on_step=False, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity, reconstructions)

        if self.conf.training.use_ema:
            self.ema.restore(self.parameters())
        return mse

    def test_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('depth_vel')

        ldm_cond_embedding = self.ldm_cond_encoder(batch)['s16']
        del batch

        # 2. 模型
        recon_z = self.ldm.sample(cond=ldm_cond_embedding, x_size=(depth_velocity.shape[0], 16, 16, 16))
        reconstructions = self.vae.decode(recon_z)

        # 3. 损失
        mse = F.mse_loss(depth_velocity, reconstructions)
        mae = F.l1_loss(depth_velocity, reconstructions)
        self.log('test/mse', mse.detach().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mae', mae.detach().item(), on_step=False, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        self.test_metrics.update(reconstructions, depth_velocity)
        self._last_test_batch = (depth_velocity, reconstructions)
        self.save_batch_torch(batch_idx, reconstructions, save_dir=self.conf.testing.test_save_dir)
        return mse
