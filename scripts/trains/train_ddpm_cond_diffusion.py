from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning
from scripts.trains.basetrain import base_train, base_train_field_cut


def train_ddpm_cond_diffusion():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/ddpm_cond_diffusion.yaml')
    # 获取当前日期时间
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d-%H")
    conf.training.logging.log_version = date_str + "base-cond_rms-smooth_All-data_T-200-10"

    model = DDPMConditionalDiffusionLightning(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False, )


def fine_tuning_ddpm_cond_diffusion():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/ddpm_cond_diffusion.yaml')
    conf.testing.ckpt_path = 'logs/ddpm_diffusion/tensorboard/base_cond-CurveVelA-1124/checkpoints/epoch_43-loss0.926.ckpt'
    conf.training.logging.log_version = "fine_tuning_field_cut_normalize"
    conf.testing.test_save_dir = 'logs/ddpm_diffusion/test_results/field_cut1229'
    conf.training.lr = 1e-6
    conf.training.max_epochs = 50
    conf.training.min_epochs = 0
    conf.training.dataloader.batch_size = 4

    model = DDPMConditionalDiffusionLightning.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)

    base_train_field_cut(model, conf, fast_run=True, use_lr_finder=False)


if __name__ == '__main__':
    train_ddpm_cond_diffusion()
    # fine_tuning_ddpm_cond_diffusion()
