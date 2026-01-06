import torch
from omegaconf import OmegaConf

from lightning_modules.baselines_lightning.VelocityGANLightning import VelocityGANLightning
from scripts.test.basetest import base_test


def test_velocity_gan(dataset_name, ckpt_path):
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/velocity_gan.yaml')
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/{dataset_name}'
    conf.testing.ckpt_path = ckpt_path
    conf.datasets.dataset_name[0] = dataset_name
    conf.training.logging.log_version = dataset_name + "_test"

    model = VelocityGANLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_test(model, conf)


if __name__ == '__main__':
    ckpt_paths = {'CurveVelA': 'logs/velocity_gan/tensorboard/CurveVelA1003_final/checkpoints/epoch_72-ssim_0.617.ckpt',
        'FlatVelA': 'logs/velocity_gan/tensorboard/FlatVelA1004_final/checkpoints/epoch_70-ssim_0.690.ckpt',
        'FlatVelB': 'logs/velocity_gan/tensorboard/FlatVelB1005_final/checkpoints/epoch_81-ssim_0.407.ckpt',
        'CurveVelB': 'logs/velocity_gan/tensorboard/CurveVelB1006_final/checkpoints/epoch_93-ssim_0.366.ckpt', }
    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        print(f"------------------{dataset_name}------------------")
        test_velocity_gan(dataset_name, ckpt_paths[dataset_name])
