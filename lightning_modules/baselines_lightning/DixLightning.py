import torch.nn.functional as F

from lightning_modules.base_lightning import BaseLightningModule
from models.baselines.dix import SmoothDix


class DixLightning(BaseLightningModule):
    def __init__(self, batch_size, lr=2e-5, test_image_save_dir='./logs/dix/'):
        super().__init__(batch_size, lr, data_range=2.0)
        self.dix = SmoothDix()
        self.vmax = 4500.0
        self.vmin = 1500.0
        self.test_image_save_dir = test_image_save_dir

    def test_step(self, batch, batch_idx):
        # 1. 数据
        depth_vel = batch.pop('depth_vel')  # 不做归一化
        rms_vel = batch['rms_vel']
        del batch
        recon, _ = self.dix(rms_vel)
        # 3. 损失
        depth_vel = ((depth_vel - self.vmin) / (self.vmax - self.vmin)) * 2 - 1.0  # [-1, 1]
        recon = ((recon - self.vmin) / (self.vmax - self.vmin)) * 2 - 1.0  # [-1, 1]
        mse = F.mse_loss(depth_vel, recon)
        mae = F.l1_loss(depth_vel, recon)
        self.log('test/mse', mse.detach().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mae', mae.detach().item(), on_step=False, on_epoch=True, prog_bar=True)

        # 4. 评价指标
        self.test_metrics.update(recon, depth_vel)
        self._last_test_batch = (depth_vel, recon)
        self.save_batch_torch(batch_idx, recon, save_dir=self.test_image_save_dir)
        return mse


if __name__ == '__main__':
    for dataset_name in ('FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB', 'CurveFaultA'):
        print(f'\n\n\n{dataset_name}')
