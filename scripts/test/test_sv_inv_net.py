import torch
from omegaconf import OmegaConf

from lightning_modules.baselines_lightning.SVInvNetLightning import SVInvNetLightning
from scripts.test.basetest import base_test


def test_sv_inv_net(dataset_name, ckpt_path):
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/sv_inv_net.yaml')
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/{dataset_name}'
    conf.testing.ckpt_path = ckpt_path
    conf.datasets.dataset_name[0] = dataset_name
    conf.training.logging.log_version = dataset_name + "_test"

    model = SVInvNetLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_test(model, conf)


if __name__ == '__main__':
    ckpt_paths = {'CurveVelA': 'logs/sv_inv_net/tensorboard/CurveVelA1002_final/checkpoints/epoch_47-ssim0.717.ckpt',
        'FlatVelA': 'logs/sv_inv_net/tensorboard/FlatVelA1003_final/checkpoints/epoch_49-ssim0.760.ckpt',
        'FlatVelB': 'logs/sv_inv_net/tensorboard/FlatVelB1004_final/checkpoints/epoch_99-ssim0.828.ckpt',
        'CurveVelB': 'logs/sv_inv_net/tensorboard/CurveVelB1005_final/checkpoints/epoch_49-ssim0.544.ckpt', }
    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        test_sv_inv_net(dataset_name, ckpt_paths[dataset_name])
