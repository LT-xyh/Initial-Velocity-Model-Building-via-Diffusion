import torch
from omegaconf import OmegaConf

from lightning_modules.diffusion.CondLatentDiffusionLightning import CondLatentDiffusionLightning
from lightning_modules.diffusion.LatentConditionalDiffusionLightning import LatentConditionalDiffusionLightning
from scripts.trains.basetrain import base_train


def train_latent_cond_diffusion():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/latent_cond_diffusion.yaml')

    # model = CondAutoencoderLightning(conf)
    # model = LatentConditionalDiffusionLightning(conf)
    model = CondLatentDiffusionLightning(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False,      )


if __name__ == '__main__':
    train_latent_cond_diffusion()
