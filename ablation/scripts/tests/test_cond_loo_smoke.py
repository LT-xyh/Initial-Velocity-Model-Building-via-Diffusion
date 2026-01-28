import torch

from models.conditional_encoder.CondFusionPyramid70 import CondFusionPyramid70


def test_cond_loo_cond_encoder_shapes_cpu():
    encoder = CondFusionPyramid70()
    batch = {
        "rms_vel": torch.randn(2, 1, 1000, 70),
        "migrated_image": torch.randn(2, 1, 1000, 70),
        "horizon": torch.randn(2, 1, 70, 70),
        "well_log": torch.randn(2, 1, 70, 70),
    }
    drop_options = [None, "rms_vel", "migrated_image", "horizon", "well_log"]
    for drop in drop_options:
        cond_batch = {k: v for k, v in batch.items()}
        if drop is not None:
            cond_batch.pop(drop, None)
        out = encoder(cond_batch)
        assert "s16" in out
        assert out["s16"].shape == (2, 64, 16, 16)
