import torch
import torch.nn.functional as F
from torch import nn

from diffusers import EMAModel
from lightning_modules.Autoencoder.CondAutoencoderLightning import MyConditionalAELightning
from lightning_modules.base_lightning import BaseLightningModule
from models.conditional_encoder.ConditionalFusion import MultiModalFusion
from models.conditional_encoder.HorizonEncoder import HorizonEncoder
from models.conditional_encoder.MigratedImagingEncoder import MigratedImagingEncoder
from models.conditional_encoder.RMSEncoder import RMSVelocityEncoder
from models.conditional_encoder.WellLogEncoder import WellLogEncoder
from models.diffusion.DiffusionConditionedUNet import LatentConditionalDiffusion


class LatentConditionalDiffusionLightning(BaseLightningModule):
    def __init__(self, conf):
        super().__init__(batch_size=conf.training.dataloader.batch_size, lr=conf.training.lr)
        self.conf = conf
        # 1. 加载预训练的autoencoder
        autoencoder = MyConditionalAELightning.load_from_checkpoint(conf.autoencoder_conf.autoencoder_checkpoint_path)
        for param in autoencoder.parameters():
            param.requires_grad = False
        self.ae = autoencoder.ae
        self.ae_cond_encoder = autoencoder.cond_encoder
        self.ae_multi_fusion = autoencoder.multi_fusion
        del autoencoder

        # 2. latent diffusion
        self.ldm = LatentConditionalDiffusion(timesteps=conf.latent_diffusion.schedule.timesteps,
                                              beta_schedule=conf.latent_diffusion.schedule.beta_schedule,
                                              model_mode=conf.latent_diffusion.cond.model_mode, token_pool=None,
                                              # crossattn 可设 2/4 降 token
                                              ddim_eta=0.0, )
        self.ldm_cond_encoder = nn.ModuleDict(
            {'rms_encoder': RMSVelocityEncoder(), 'migrated_imaging_encoder': MigratedImagingEncoder(),
             'horizon_encoder': HorizonEncoder(), 'well_log_encoder': WellLogEncoder(), })
        self.ldm_multi_fusion = MultiModalFusion(in_channels={'rms': 32, 'migrated': 64, 'horizon': 16, 'well': 16},
                                                 C_out=64,  # 统一条件嵌入通道，推荐 64
                                                 C_mid=128, return_vector_dim=0,  # 如需 AdaIN/FiLM 的全局向量；不需要可设 0
                                                 modality_dropout_p=0  # 训练期可开；推理期会自动关闭
                                                 )
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
        depth_velocity = batch['model']

        ae_cond = self.ae_multi_fusion({'rms': self.ae_cond_encoder['rms_encoder'](batch['rms_vel']),
                                        'migrated': self.ae_cond_encoder['migrated_imaging_encoder'](batch['migrate']),
                                        'horizon': self.ae_cond_encoder['horizon_encoder'](batch['horizens']),
                                        'well': self.ae_cond_encoder['well_log_encoder'](batch['well_log'])})['map']
        ldm_cond = self.ldm_multi_fusion({'rms': self.ldm_cond_encoder['rms_encoder'](batch['rms_vel']),
                                          'migrated': self.ldm_cond_encoder['migrated_imaging_encoder'](
                                              batch['migrate']),
                                          'horizon': self.ldm_cond_encoder['horizon_encoder'](batch['horizens']),
                                          'well': self.ldm_cond_encoder['well_log_encoder'](batch['well_log'])})['map']
        del batch

        # 2. 模型
        latents = self.ae.encode(depth_velocity)
        ldm_dict = self.ldm.training_loss(x0=latents, cond=ldm_cond)
        with torch.no_grad():
            recon_z = ldm_dict['pre_x0'].detach()
            reconstructions = self.ae.decode(recon_z, ae_cond)

        # 3. 损失
        loss = ldm_dict['loss']
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

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
        ae_cond = self.ae_multi_fusion({'rms': self.ae_cond_encoder['rms_encoder'](batch['rms_vel']),
                                        'migrated': self.ae_cond_encoder['migrated_imaging_encoder'](batch['migrate']),
                                        'horizon': self.ae_cond_encoder['horizon_encoder'](batch['horizens']),
                                        'well': self.ae_cond_encoder['well_log_encoder'](batch['well_log'])})['map']
        ldm_cond = self.ldm_multi_fusion({'rms': self.ldm_cond_encoder['rms_encoder'](batch['rms_vel']),
                                          'migrated': self.ldm_cond_encoder['migrated_imaging_encoder'](
                                              batch['migrate']),
                                          'horizon': self.ldm_cond_encoder['horizon_encoder'](batch['horizens']),
                                          'well': self.ldm_cond_encoder['well_log_encoder'](batch['well_log'])})['map']

        # 2. 模型
        recon_z = self.ldm.sample(cond=ldm_cond, num_inference_steps=50, eta=0.0)
        with torch.no_grad():
            reconstructions = self.ae.decode(recon_z, ae_cond)

        # 3. 损失
        with torch.no_grad():
            loss = F.mse_loss(depth_velocity, reconstructions)
        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True)

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
