from datetime import datetime

import torch
from omegaconf import OmegaConf

from ablation.lightning.DDPMConditionalDiffusionLightning_LOO import (
    DDPMConditionalDiffusionLightningLOO,
)
from scripts.trains.basetrain import base_train

# Overrides to run in one sweep (exclude full).
OVERRIDE_PATHS = (
    "ablation/configs/loo_drop_rms.yaml",
    "ablation/configs/loo_drop_pstm.yaml",
    "ablation/configs/loo_drop_hor.yaml",
    "ablation/configs/loo_drop_well.yaml",
)


def _resolve_drop_tag(conf) -> str:
    ablation_conf = conf.get("ablation", None) if hasattr(conf, "get") else None
    drop = None
    if ablation_conf is not None:
        drop = ablation_conf.get("drop_modality", None)
    if drop is None:
        return "full"
    drop_str = str(drop).strip()
    if drop_str.lower() in ("none", "null", ""):
        return "full"
    return f"drop_{drop_str}"


def train_ddpm_cond_loo(override_path: str | None = None) -> None:
    torch.set_float32_matmul_precision("medium")
    conf = OmegaConf.load("configs/ddpm_cond_diffusion.yaml")
    if override_path:
        override = OmegaConf.load(override_path)
        conf = OmegaConf.merge(conf, override)
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d-%H")
    conf.training.logging.log_version = f"{date_str}loo_{_resolve_drop_tag(conf)}"

    model = DDPMConditionalDiffusionLightningLOO(conf)
    base_train(model, conf, fast_run=False, use_lr_finder=False)


if __name__ == "__main__":
    for override in OVERRIDE_PATHS:
        train_ddpm_cond_loo(override)
