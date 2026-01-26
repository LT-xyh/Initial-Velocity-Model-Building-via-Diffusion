from datetime import datetime
import argparse
from typing import Optional

import torch
from omegaconf import OmegaConf

from ablation.lightning.DDPMConditionalDiffusionLightning_CondAblationVariants import (
    DDPMConditionalDiffusionLightningCondAblationVariants,
)
from scripts.trains.basetrain import base_train


def train_cond_ablation_variants(override_path: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("medium")
    conf = OmegaConf.load("configs/ddpm_cond_diffusion.yaml")
    if override_path:
        override = OmegaConf.load(override_path)
        conf = OmegaConf.merge(conf, override)
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d-%H")
    conf.training.logging.log_version = date_str + "ablation_naive_cond_variant"

    model = DDPMConditionalDiffusionLightningCondAblationVariants(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", type=str, default=None, help="Path to ablation override YAML.")
    args = parser.parse_args()
    train_cond_ablation_variants(args.override)
