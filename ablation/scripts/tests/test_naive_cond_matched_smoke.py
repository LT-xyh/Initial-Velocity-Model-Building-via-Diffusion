import torch

from ablation.condition_encoder.naive_resnet_cond_encoder_matched import NaiveResNetCondEncoderMatched
from ablation.utils.param_count import count_params


def test_naive_resnet_cond_encoder_matched_shapes_and_params_cpu():
    encoder = NaiveResNetCondEncoderMatched()
    batch = {
        "rms_vel": torch.randn(2, 1, 1000, 70),
        "migrated_image": torch.randn(2, 1, 1000, 70),
        "horizon": torch.randn(2, 1, 70, 70),
        "well_log": torch.randn(2, 1, 70, 70),
    }
    out = encoder(batch)
    assert out["s16"].shape == (2, 64, 16, 16)

    params = count_params(encoder)
    target = 2_520_000
    assert abs(params - target) / target <= 0.05
