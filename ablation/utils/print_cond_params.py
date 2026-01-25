import torch

from ablation.condition_encoder.naive_resnet_cond_encoder import NaiveResNetCondEncoder
from ablation.condition_encoder.naive_resnet_cond_encoder_matched import NaiveResNetCondEncoderMatched
from ablation.utils.param_count import count_params


def _dummy_batch(batch_size: int = 2) -> dict:
    return {
        "rms_vel": torch.randn(batch_size, 1, 1000, 70),
        "migrated_image": torch.randn(batch_size, 1, 1000, 70),
        "horizon": torch.randn(batch_size, 1, 70, 70),
        "well_log": torch.randn(batch_size, 1, 70, 70),
    }


def main() -> None:
    dummy = _dummy_batch()

    small = NaiveResNetCondEncoder()
    _ = small(dummy)
    small_params = count_params(small)

    matched = NaiveResNetCondEncoderMatched()
    matched_params = count_params(matched)

    print(f"Naive-Small params: {small_params}")
    print(f"Naive-Matched params: {matched_params}")


if __name__ == "__main__":
    main()
