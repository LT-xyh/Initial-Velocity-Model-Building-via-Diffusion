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
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/test/fault_{date_str}'
    conf.testing.ckpt_path = 'logs/baselines/inversion_net/tensorboard/260118-17_All-Data_lr-1e-3/checkpoints/epoch_0-ssim0.555.ckpt'
    conf.training.logging.log_version = f"test/fault__{date_str}"
    model = InversionNetLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_fault_test(model, conf, fast_run=False)


def test_inversion_net(dataset_name):
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d")
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/inversion_net.yaml')
    conf.datasets.dataset_name = [dataset_name, ]
    print(conf.testing.test_save_dir)
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/test/{date_str}/{dataset_name}'
    conf.testing.ckpt_path = 'logs/baselines/inversion_net/tensorboard/260118-17_All-Data_lr-1e-3/checkpoints/epoch_0-ssim0.555.ckpt'
    conf.training.logging.log_version = f"test/{date_str}_{dataset_name}"
    model = InversionNetLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_test(model, conf, fast_run=False)


if __name__ == '__main__':

    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        print(f'\n\n{dataset_name}\n')
        test_inversion_net(dataset_name)
        # break

    print('\n\nFaultVelA\n')
    test_fault_inversion_net()
