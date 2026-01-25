from datetime import datetime
import argparse

import torch
from omegaconf import OmegaConf

from lightning_modules.Autoencoder.autoencoder_kl_lightning import AutoencoderKLLightning
from scripts.trains.basetrain import base_train


def _get_conf_path(latent_hw: int) -> str:
    if latent_hw == 8:
        return "ablation/configs/autoencoder_kl_latent_hw8.yaml"
    if latent_hw == 32:
        return "ablation/configs/autoencoder_kl_latent_hw32.yaml"
    raise ValueError(f"Unsupported latent_hw={latent_hw}. Expected 8 or 32.")


def train_autoencoder_kl_latent_hw(latent_hw: int):
    torch.set_float32_matmul_precision('medium')
    conf = OmegaConf.load(_get_conf_path(latent_hw))
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d")
    conf.training.logging.log_version = f"latent_hw{latent_hw}_{date_str}"

    model = AutoencoderKLLightning(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train AutoencoderKL latent spatial size ablations.")
    parser.add_argument("--latent_hw", type=int, choices=[8, 32], required=True,
                        help="Target latent spatial size (H=W).")
    return parser.parse_args()


if __name__ == "__main__":
    train_autoencoder_kl_latent_hw(32)  # 8 or 32
