from datetime import datetime

import torch
from omegaconf import OmegaConf

from ablation.lightning.DDPMConditionalDiffusionLightning_CondAblation import (
    DDPMConditionalDiffusionLightningCondAblation,
)
from scripts.trains.basetrain import base_train


def train_ddpm_cond_ablation():
    torch.set_float32_matmul_precision("medium")
    conf = OmegaConf.load("configs/ddpm_cond_diffusion.yaml")
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d-%H")
    conf.training.logging.log_version = date_str + "ablation_naive_resnet_cond"

    model = DDPMConditionalDiffusionLightningCondAblation(conf)
    base_train(model, conf, fast_run=True, use_lr_finder=False)


if __name__ == "__main__":
    train_ddpm_cond_ablation()
