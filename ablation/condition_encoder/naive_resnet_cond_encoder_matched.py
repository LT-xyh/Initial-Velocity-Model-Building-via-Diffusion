from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, channels: int, groups: int) -> nn.Module:
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        num_groups = min(groups, channels)
        while num_groups > 1 and channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class _ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_type: str,
        groups: int,
        bottleneck_expansion: int,
    ) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.bottleneck_expansion = bottleneck_expansion

        if bottleneck_expansion == 1:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.norm1 = _make_norm(norm_type, out_channels, groups)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm2 = _make_norm(norm_type, out_channels, groups)
            self.conv3 = None
            self.norm3 = None
        else:
            if out_channels % bottleneck_expansion != 0:
                raise ValueError("out_channels must be divisible by bottleneck_expansion.")
            mid_channels = out_channels // bottleneck_expansion
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
            self.norm1 = _make_norm(norm_type, mid_channels, groups)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.norm2 = _make_norm(norm_type, mid_channels, groups)
            self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm3 = _make_norm(norm_type, out_channels, groups)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_norm(norm_type, out_channels, groups),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        if self.conv3 is not None:
            out = self.norm3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.act(out + identity)
        return out


def _make_stage(
    in_channels: int,
    out_channels: int,
    blocks: int,
    stride: int,
    norm_type: str,
    groups: int,
    bottleneck_expansion: int,
) -> nn.Sequential:
    layers = [
        _ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm_type=norm_type,
            groups=groups,
            bottleneck_expansion=bottleneck_expansion,
        )
    ]
    for _ in range(1, blocks):
        layers.append(
            _ResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                norm_type=norm_type,
                groups=groups,
                bottleneck_expansion=bottleneck_expansion,
            )
        )
    return nn.Sequential(*layers)


class NaiveResNetCondEncoderMatched(nn.Module):
    def __init__(
        self,
        in_channels_by_key: Optional[Dict[str, int]] = None,
        base_channels: int = 85,
        blocks_per_stage: Tuple[int, int] = (4, 4),
        channel_multipliers: Tuple[int, int] = (1, 2),
        bottleneck_expansion: int = 1,
        out_channels: int = 64,
        norm_type: str = "gn",
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.modality_keys = ("rms_vel", "migrated_image", "horizon", "well_log")
        if in_channels_by_key is None:
            in_channels_by_key = {k: 1 for k in self.modality_keys}
        self.in_channels_by_key = {k: int(in_channels_by_key.get(k, 0)) for k in self.modality_keys}
        if any(v <= 0 for v in self.in_channels_by_key.values()):
            raise ValueError("All modalities must have positive in_channels_by_key values.")

        if len(blocks_per_stage) != 2:
            raise ValueError("blocks_per_stage must have length 2.")
        if len(channel_multipliers) != 2:
            raise ValueError("channel_multipliers must have length 2.")
        if bottleneck_expansion < 1:
            raise ValueError("bottleneck_expansion must be >= 1.")

        self.base_channels = int(base_channels)
        self.blocks_per_stage = tuple(int(x) for x in blocks_per_stage)
        self.channel_multipliers = tuple(int(x) for x in channel_multipliers)
        self.bottleneck_expansion = int(bottleneck_expansion)
        self.out_channels = int(out_channels)
        self.norm_type = norm_type
        self.gn_groups = int(gn_groups)

        in_channels_total = sum(self.in_channels_by_key.values())
        stage1_channels = self.base_channels * self.channel_multipliers[0]
        stage2_channels = self.base_channels * self.channel_multipliers[1]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels_total, self.base_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(self.norm_type, self.base_channels, self.gn_groups),
            nn.GELU(),
        )
        self.stage1 = _make_stage(
            in_channels=self.base_channels,
            out_channels=stage1_channels,
            blocks=self.blocks_per_stage[0],
            stride=2,
            norm_type=self.norm_type,
            groups=self.gn_groups,
            bottleneck_expansion=self.bottleneck_expansion,
        )
        self.stage2 = _make_stage(
            in_channels=stage1_channels,
            out_channels=stage2_channels,
            blocks=self.blocks_per_stage[1],
            stride=2,
            norm_type=self.norm_type,
            groups=self.gn_groups,
            bottleneck_expansion=self.bottleneck_expansion,
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.head = nn.Conv2d(stage2_channels, self.out_channels, kernel_size=1, bias=True)

    @staticmethod
    def _to_2d70(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {tuple(x.shape)}")
        h, w = x.shape[-2], x.shape[-1]
        if h == 70 and w == 70:
            return x
        if w == 70 and h != 70:
            return F.adaptive_avg_pool2d(x, (70, 70))
        return F.interpolate(x, size=(70, 70), mode="bilinear", align_corners=False)

    def _collect_modalities(self, cond_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        tensors: Dict[str, torch.Tensor] = {}
        ref_tensor = None
        for key in self.modality_keys:
            value = cond_dict.get(key, None)
            if torch.is_tensor(value):
                value = self._to_2d70(value)
                expected_c = self.in_channels_by_key[key]
                if value.shape[1] != expected_c:
                    raise ValueError(
                        f"Unexpected channels for {key}: got {value.shape[1]}, expected {expected_c}"
                    )
                tensors[key] = value
                if ref_tensor is None:
                    ref_tensor = value
        if ref_tensor is None:
            raise ValueError("No valid condition tensors found in cond_dict.")

        for key in self.modality_keys:
            if key not in tensors:
                c = self.in_channels_by_key[key]
                tensors[key] = torch.zeros(
                    (ref_tensor.shape[0], c, 70, 70),
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                )
        return torch.cat([tensors[key] for key in self.modality_keys], dim=1)

    def forward(self, cond_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(cond_dict, dict):
            raise TypeError(f"cond_dict must be a dict, got {type(cond_dict)}")
        x = self._collect_modalities(cond_dict)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = self.head(x)
        return {"s16": x}
