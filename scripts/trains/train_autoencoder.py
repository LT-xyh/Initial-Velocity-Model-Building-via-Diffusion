import torch
from omegaconf import OmegaConf

from lightning_modules.Autoencoder.autoencoder_kl_lightning import AutoencoderAELightning
from lightning_modules.Autoencoder.AutoencoderLightning import AutoencoderLightning
from scripts.trains.basetrain import base_train


def train_autoencoder():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/autoencoder.yaml')

    model = AutoencoderAELightning(conf)

    base_train(model, conf, fast_run=False, use_lr_finder=False)

def train_my_autoencoder():
    torch.set_float32_matmul_precision('medium')  # 设置矩阵乘法精度利用Tensor Cores
    conf = OmegaConf.load('configs/autoencoder.yaml')
    model = AutoencoderLightning(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False)

if __name__ == '__main__':
    # train_autoencoder()
    train_my_autoencoder()