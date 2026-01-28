from datetime import datetime

import torch
from omegaconf import OmegaConf

from ablation.lightning.DDPMConditionalDiffusionLightning_LOO import (
    DDPMConditionalDiffusionLightningLOO,
)
from scripts.test.basetest import base_test

# Overrides to run in one sweep (exclude full).
OVERRIDE_PATHS = (
    "ablation/configs/loo_drop_rms.yaml",
    "ablation/configs/loo_drop_pstm.yaml",
    "ablation/configs/loo_drop_hor.yaml",
    "ablation/configs/loo_drop_well.yaml",
)

# Per-modality checkpoints for ablation runs.
# Update these paths to the actual checkpoints produced by training.
CKPT_PATHS = {
    "rms_vel": "",
    "migrated_image": "",
    "horizon": "",
    "well_log": "",
}


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


def test_ddpm_cond_loo(override_path: str | None = None) -> None:
    torch.set_float32_matmul_precision("medium")
    conf = OmegaConf.load("configs/ddpm_cond_diffusion.yaml")
    if override_path:
        override = OmegaConf.load(override_path)
        conf = OmegaConf.merge(conf, override)
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d-%H")
    conf.training.logging.log_version = f"{date_str}loo_{_resolve_drop_tag(conf)}"

    drop_tag = _resolve_drop_tag(conf)
    if drop_tag.startswith("drop_"):
        drop_key = drop_tag.replace("drop_", "", 1)
    else:
        drop_key = None
    if drop_key is not None:
        ckpt_path = CKPT_PATHS.get(drop_key, "")
        if not ckpt_path:
            raise ValueError(f"Missing CKPT_PATHS entry for drop '{drop_key}'.")
        conf.testing.ckpt_path = ckpt_path
    if not conf.testing.ckpt_path:
        raise ValueError("conf.testing.ckpt_path is empty; set a checkpoint path before testing.")

    model = DDPMConditionalDiffusionLightningLOO.load_from_checkpoint(conf.testing.ckpt_path, conf=conf)
    base_test(model, conf, fast_run=False)


if __name__ == "__main__":
    for override in OVERRIDE_PATHS:
        test_ddpm_cond_loo(override)
