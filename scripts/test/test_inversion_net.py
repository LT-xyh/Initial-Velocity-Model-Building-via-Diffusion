from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.baselines_lightning.InversionNetLightning import InversionNetLightning
from scripts.test.basetest import base_test, base_fault_test


def test_fault_inversion_net():
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d")
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/inversion_net.yaml')
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/test_fault_{date_str}'
    conf.testing.ckpt_path = 'logs/baselines/inversion_net/tensorboard/1231_/checkpoints/epoch_44-ssim0.659.ckpt'
    conf.training.logging.log_version = f"test_fault__{date_str}"
    model = InversionNetLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_fault_test(model, conf, fast_run=False)


if __name__ == '__main__':
    test_fault_inversion_net()
