import torch

from ablation.condition_encoder.naive_resnet_cond_encoder import NaiveResNetCondEncoder


def test_naive_resnet_cond_encoder_shapes_cpu():
    encoder = NaiveResNetCondEncoder()
    batch = {
        "rms_vel": torch.randn(2, 1, 1000, 70),
        "migrated_image": torch.randn(2, 1, 1000, 70),
        "horizon": torch.randn(2, 1, 70, 70),
        "well_log": torch.randn(2, 1, 70, 70),
    }
    out = encoder(batch)
    assert "s16" in out
    assert out["s16"].shape == (2, 64, 16, 16)
