from typing import Iterable, Tuple

from ablation.condition_encoder.naive_resnet_cond_encoder_matched import NaiveResNetCondEncoderMatched
from ablation.utils.param_count import count_params


def _grid() -> Iterable[Tuple[int, Tuple[int, int], Tuple[int, int], int]]:
    base_channels_options = [64, 72, 80, 84, 85, 86, 88, 96, 104, 112]
    blocks_options = [(3, 3), (4, 4), (5, 5), (4, 5)]
    multipliers_options = [(1, 2), (1, 3), (2, 2)]
    bottleneck_options = [1]
    for base in base_channels_options:
        for blocks in blocks_options:
            for mult in multipliers_options:
                for bottleneck in bottleneck_options:
                    yield base, blocks, mult, bottleneck


def search(target_params: int = 2_520_000, tolerance: float = 0.03) -> None:
    best = None
    best_diff = None
    for base, blocks, mult, bottleneck in _grid():
        model = NaiveResNetCondEncoderMatched(
            base_channels=base,
            blocks_per_stage=blocks,
            channel_multipliers=mult,
            bottleneck_expansion=bottleneck,
        )
        params = count_params(model)
        diff = abs(params - target_params) / target_params
        if best_diff is None or diff < best_diff:
            best = (base, blocks, mult, bottleneck, params, diff)
            best_diff = diff
        if diff <= tolerance:
            print(
                f"Hit: base={base} blocks={blocks} mult={mult} bottleneck={bottleneck} "
                f"params={params} diff={diff:.4f}"
            )
            return
    if best is not None:
        base, blocks, mult, bottleneck, params, diff = best
        print(
            f"Best: base={base} blocks={blocks} mult={mult} bottleneck={bottleneck} "
            f"params={params} diff={diff:.4f}"
        )


if __name__ == "__main__":
    search()
