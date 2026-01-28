from typing import Optional

import torch
import torch.nn.functional as F

from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning


class DDPMConditionalDiffusionLightningLOO(DDPMConditionalDiffusionLightning):
    def __init__(self, conf):
        super().__init__(conf)
        drop_modality = None
        ablation_conf = conf.get("ablation", None) if hasattr(conf, "get") else None
        if ablation_conf is not None:
            drop_modality = ablation_conf.get("drop_modality", None)
        if isinstance(drop_modality, str):
            normalized = drop_modality.strip()
            if normalized.lower() in ("none", "null", ""):
                drop_modality = None
            else:
                drop_modality = normalized
        self.drop_modality = drop_modality
        self.save_hyperparameters({"ablation_drop_modality": self.drop_modality})

    def training_step(self, batch, batch_idx):
        cond_batch = dict(batch)
        depth_velocity = cond_batch.pop("depth_vel")

        if self.drop_modality is not None and self.drop_modality in cond_batch:
            del cond_batch[self.drop_modality]
        ldm_cond_embedding = self.ldm_cond_encoder(cond_batch)["s16"]
        del cond_batch

        posterior = self.vae.encode(depth_velocity)
        latents = posterior.sample()
        ldm_dict = self.ldm.training_loss(x0=latents, cond=ldm_cond_embedding, loss_type="mse")
        with torch.no_grad():
            recon_z = ldm_dict["x0_pred"].detach()
            reconstructions = self.vae.decode(recon_z)

        loss = ldm_dict["loss"]
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)

        if self.conf.training.use_ema:
            self.ema.step(self._ema_params())

        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log("train/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            self.train_metrics.update(depth_velocity, reconstructions)

        return loss

    def validation_step(self, batch, batch_idx):
        cond_batch = dict(batch)
        depth_velocity = cond_batch.pop("depth_vel")

        if self.drop_modality is not None and self.drop_modality in cond_batch:
            del cond_batch[self.drop_modality]
        ldm_cond_embedding = self.ldm_cond_encoder(cond_batch)["s16"]
        del cond_batch

        recon_z = self.ldm.sample(cond=ldm_cond_embedding, x_size=(depth_velocity.shape[0], 16, 16, 16))
        with torch.no_grad():
            reconstructions = self.vae.decode(recon_z)

        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log("val/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            self.val_metrics.update(reconstructions, depth_velocity)
        self._last_val_batch = (depth_velocity.detach(), reconstructions.detach())

        return mse

    def test_step(self, batch, batch_idx):
        cond_batch = dict(batch)
        depth_velocity = cond_batch.pop("depth_vel")
        well_log = cond_batch.get("well_log", None)

        if self.drop_modality is not None and self.drop_modality in cond_batch:
            del cond_batch[self.drop_modality]
        ldm_cond_embedding = self.ldm_cond_encoder(cond_batch)["s16"]

        recon_z = self.ldm.sample(cond=ldm_cond_embedding, x_size=(depth_velocity.shape[0], 16, 16, 16))
        with torch.no_grad():
            reconstructions = self.vae.decode(recon_z)

        with torch.no_grad():
            mse = F.mse_loss(depth_velocity, reconstructions)
            mae = F.l1_loss(depth_velocity, reconstructions)
        self.log("test/mse", mse.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", mae.detach(), on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            self.test_metrics.update(reconstructions, depth_velocity)
            well_metrics = self.well_match_metrics(reconstructions, depth_velocity, well_log)
            if well_metrics is not None:
                self.log("test/well_mae", well_metrics["well_mae"].detach(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("test/well_mse", well_metrics["well_mse"].detach(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("test/well_cc", well_metrics["well_cc"].detach(), on_step=False, on_epoch=True, prog_bar=True)
        self._last_test_batch = (depth_velocity.detach(), reconstructions.detach())
        if batch_idx < 2:
            self.save_batch_torch(batch_idx, reconstructions, save_dir=f"{self.conf.testing.test_save_dir}",
                                  well_log=well_log)

        return mse
