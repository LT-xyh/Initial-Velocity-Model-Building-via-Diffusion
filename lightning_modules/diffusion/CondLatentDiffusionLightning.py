import torch
import torch.nn.functional as F

from diffusers import EMAModel
from lightning_modules.Autoencoder.CondUnetAutoencoderLightning import CondUnetAutoencoderZLightning
from lightning_modules.base_lightning import BaseLightningModule
from models.diffusion.DiffusionConditionedUNet import LatentConditionalDiffusion


class CondLatentDiffusionLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf
        # 1. 加载预训练的autoencoder
        # autoencoder = CondUnetAutoencoderLightning.load_from_checkpoint(
        # conf.autoencoder_conf.autoencoder_checkpoint_path)
        autoencoder = CondUnetAutoencoderZLightning.load_from_checkpoint(
            conf.autoencoder_conf.autoencoder_checkpoint_path)
        for param in autoencoder.parameters():
            param.requires_grad = False
        self.ae = autoencoder.ae
        self.cond_encoder = autoencoder.cond_encoder
        # self.ldm_cond_encoder = self.cond_encoder
        self.ldm_cond_encoder = autoencoder.cond_encoder
        for param in self.ldm_cond_encoder.parameters():
            param.requires_grad = True
        del autoencoder

        # 2. latent diffusion
        self.ldm = LatentConditionalDiffusion()
        if self.conf.training.use_ema:
            self._ema_parameters = [p for p in self.parameters() if p.requires_grad]

    def setup(self, stage):
        super().setup(stage)
        if self.conf.training.use_ema:
            if self.ema is None:
                self.ema = EMAModel(parameters=self._ema_parameters, use_ema_warmup=True, foreach=True, power=0.75,
                                    device=self.device)

    def training_step(self, batch, batch_idx):
        # 1. 数据
        depth_velocity = batch.pop('model')
        depth_velocity = self.normalize_to_neg_one_to_one(depth_velocity)

        cond_embedding = self.cond_encoder(batch)
        ldm_cond_embedding = self.ldm_cond_encoder(batch)['s16']
        del batch

        # 2. 模型
        latents = self.ae.encode(depth_velocity)
        ldm_dict = self.ldm.training_loss(x0=latents, cond=ldm_cond_embedding, loss_type="mse")
        with torch.no_grad():
            recon_z = ldm_dict['pre_x0'].detach()
            reconstructions = self.ae.decode(recon_z, cond_embedding)

        # 3. 损失
        loss = ldm_dict['loss']
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        depth_velocity = self.unnormalize_from_neg_one_to_one(depth_velocity)
        reconstructions = self.unnormalize_from_neg_one_to_one(reconstructions)
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
        ldm_cond_embedding = self.ldm_cond_encoder(batch)['s16']
        del batch

        # 2. 模型
        recon_z = self.ldm.sample(cond=ldm_cond_embedding, x_size=(depth_velocity.shape[0], 16, 16, 16))
        with torch.no_grad():
            reconstructions = self.ae.decode(recon_z, cond_embedding)

        # 3. 损失
        depth_velocity = self.unnormalize_from_neg_one_to_one(depth_velocity)
        reconstructions = self.unnormalize_from_neg_one_to_one(reconstructions)
        with torch.no_grad():
            loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('val/mse', loss.detach(), on_step=False, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        self.val_metrics.update(depth_velocity, reconstructions)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        if self.conf.training.use_ema:
            self.ema.restore(self._ema_params())
        return loss


if __name__ == "__main__":
    def test_checkpoint():
        # 仅查看检查点内容（不加载到模型）
        checkpoint = torch.load(
            'logs/autoencoder/my_cond_autoencoder/tensorboard/CurveVelA_ema/checkpoints/epoch_16-loss0.0022.ckpt',
            weights_only=False)
        print(checkpoint.keys())  # 查看检查点包含的键（如model_state_dict、optimizer_state_dict等）
        print(checkpoint['epoch'])  # 查看保存时的 epoch
        # 查看模型参数的键（网络层名称）
        print(checkpoint['model_state_dict'].keys())


    test_checkpoint()
