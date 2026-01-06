import torch
from omegaconf import OmegaConf

from lightning_modules.baselines_lightning.SVInvNetLightning import MAblationSVInvNetLightning, \
    MVAblationSVInvNetLightning, MHAblationSVInvNetLightning, MWAblationSVInvNetLightning
from scripts.test.basetest import base_test


def test_ablations(dataset_name, ckpt_path):
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/sv_inv_net.yaml')
    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/{dataset_name}'
    conf.training.logging.log_dir = 'logs/ablations/sv_inv_net/test'
    conf.datasets.dataset_name[0] = dataset_name
    ablation_dict = {
        'M': MAblationSVInvNetLightning,
        'MV': MVAblationSVInvNetLightning,
        'MH': MHAblationSVInvNetLightning,
        'MW': MWAblationSVInvNetLightning,
    }
    for ab in ablation_dict.keys():
        print(f"\n\n\n----------------------{dataset_name}----{ab}----------------------------")
        conf.testing.ckpt_path = ckpt_path[ab]
        conf.training.logging.log_version = dataset_name + "_ablation_" + ab + "_test"
        model = ablation_dict[ab].load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
        base_test(model, conf)


if __name__ == '__main__':
    ckpt_paths = {
        'CurveVelA': {
            'M': 'logs/ablations/sv_inv_net/tensorboard/CurveVelA_ablation_M/checkpoints/epoch_49-ssim0.402.ckpt',
            'MV': 'logs/ablations/sv_inv_net/tensorboard/CurveVelA_ablation_MV/checkpoints/epoch_42-ssim0.638.ckpt',
            'MH': 'logs/ablations/sv_inv_net/tensorboard/CurveVelA_ablation_MH/checkpoints/epoch_48-ssim0.652.ckpt',
            'MW': 'logs/ablations/sv_inv_net/tensorboard/CurveVelA_ablation_MW/checkpoints/epoch_42-ssim0.461.ckpt',
        },
        'FlatVelA': {
            'M': 'logs/ablations/sv_inv_net/tensorboard/FlatVelA_ablation_M/checkpoints/epoch_48-ssim0.383.ckpt',
            'MV': 'logs/ablations/sv_inv_net/tensorboard/FlatVelA_ablation_MV/checkpoints/epoch_45-ssim0.616.ckpt',
            'MH': 'logs/ablations/sv_inv_net/tensorboard/FlatVelA_ablation_MH/checkpoints/epoch_28-ssim0.505.ckpt',
            'MW': 'logs/ablations/sv_inv_net/tensorboard/FlatVelA_ablation_MW/checkpoints/epoch_29-ssim0.410.ckpt',
        },
        'FlatVelB': {
            'M': 'logs/ablations/sv_inv_net/tensorboard/FlatVelB_ablation_M/checkpoints/epoch_44-ssim0.298.ckpt',
            'MV': 'logs/ablations/sv_inv_net/tensorboard/FlatVelB_ablation_MV/checkpoints/epoch_49-ssim0.356.ckpt',
            'MH': 'logs/ablations/sv_inv_net/tensorboard/FlatVelB_ablation_MH/checkpoints/epoch_49-ssim0.339.ckpt',
            'MW': 'logs/ablations/sv_inv_net/tensorboard/FlatVelB_ablation_MW/checkpoints/epoch_49-ssim0.594.ckpt',
        },
        'CurveVelB': {
            'M': 'logs/ablations/sv_inv_net/tensorboard/CurveVelB_ablation_M/checkpoints/epoch_48-ssim0.266.ckpt',
            'MV': 'logs/ablations/sv_inv_net/tensorboard/CurveVelB_ablation_MV/checkpoints/epoch_31-ssim0.331.ckpt',
            'MH': 'logs/ablations/sv_inv_net/tensorboard/CurveVelB_ablation_MH/checkpoints/epoch_25-ssim0.235.ckpt',
            'MW': 'logs/ablations/sv_inv_net/tensorboard/CurveVelB_ablation_MW/checkpoints/epoch_25-ssim0.335.ckpt',
        },
    }
    for dataset_name in ['CurveVelA', 'FlatVelA', 'FlatVelB', 'CurveVelB']:
        test_ablations(dataset_name, ckpt_paths[dataset_name])
