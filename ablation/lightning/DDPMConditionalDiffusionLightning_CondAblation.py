from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning

from ablation.condition_encoder.naive_resnet_cond_encoder import NaiveResNetCondEncoder


class DDPMConditionalDiffusionLightningCondAblation(DDPMConditionalDiffusionLightning):
    def __init__(self, conf):
        super().__init__(conf)
        self.ldm_cond_encoder = NaiveResNetCondEncoder()
