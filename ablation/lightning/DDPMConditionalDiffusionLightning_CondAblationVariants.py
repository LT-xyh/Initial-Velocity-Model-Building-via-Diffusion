from typing import Any, Dict, Optional

from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning

from ablation.condition_encoder.naive_resnet_cond_encoder import NaiveResNetCondEncoder
from ablation.condition_encoder.naive_resnet_cond_encoder_matched import NaiveResNetCondEncoderMatched


class DDPMConditionalDiffusionLightningCondAblationVariants(DDPMConditionalDiffusionLightning):
    def __init__(self, conf, cond_variant: Optional[str] = None):
        super().__init__(conf)
        variant = cond_variant
        if variant is None:
            ablation_conf = conf.get("ablation", None) if hasattr(conf, "get") else None
            variant = ablation_conf.get("cond_variant", "small") if ablation_conf is not None else "small"
        variant = str(variant).lower()

        if variant == "small":
            self.ldm_cond_encoder = NaiveResNetCondEncoder()
            return

        if variant == "matched":
            ablation_conf = conf.get("ablation", None) if hasattr(conf, "get") else None
            cond_conf: Dict[str, Any] = {}
            if ablation_conf is not None and ablation_conf.get("cond_encoder", None) is not None:
                cond_conf = {k: v for k, v in ablation_conf.get("cond_encoder").items()}
            for key in ("blocks_per_stage", "channel_multipliers"):
                if key in cond_conf and isinstance(cond_conf[key], list):
                    cond_conf[key] = tuple(cond_conf[key])
            self.ldm_cond_encoder = NaiveResNetCondEncoderMatched(**cond_conf)
            return

        raise ValueError(f"Unknown cond_variant: {variant}")
