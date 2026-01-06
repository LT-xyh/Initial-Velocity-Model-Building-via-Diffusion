import torch
from omegaconf import OmegaConf

from lightning_modules.Autoencoder.CondUnetAutoencoderLightning import CondUnetAutoencoderLightning, \
    CondUnetAutoencoderZLightning
from scripts.trains.basetrain import base_train


def train_cond_autoencoder():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/cond_autoencoder.yaml')

    # model = CondAutoencoderLightning(conf)
    # model = MyConditionalAELightning(conf)
    # model = CondUnetAutoencoderLightning(conf)
    model = CondUnetAutoencoderZLightning(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False)


if __name__ == '__main__':
    train_cond_autoencoder()
