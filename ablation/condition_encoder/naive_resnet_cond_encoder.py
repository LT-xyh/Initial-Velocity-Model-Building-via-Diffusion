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
    def __init__(self, channels: int, stride: int, norm_type: str, groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _make_norm(norm_type, channels, groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = _make_norm(norm_type, channels, groups)
        self.act = nn.GELU()
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                _make_norm(norm_type, channels, groups),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.act(out + identity)
        return out


class NaiveResNetCondEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        out_channels: int = 64,
        norm_type: str = "gn",
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.gn_groups = gn_groups
        self._built = False

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

    def _build(self, in_channels: int, device: torch.device) -> None:
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.base_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(self.norm_type, self.base_channels, self.gn_groups),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            _ResBlock(self.base_channels, stride=2, norm_type=self.norm_type, groups=self.gn_groups),
            _ResBlock(self.base_channels, stride=1, norm_type=self.norm_type, groups=self.gn_groups),
        )
        self.stage2 = nn.Sequential(
            _ResBlock(self.base_channels, stride=2, norm_type=self.norm_type, groups=self.gn_groups),
            _ResBlock(self.base_channels, stride=1, norm_type=self.norm_type, groups=self.gn_groups),
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.head = nn.Conv2d(self.base_channels, self.out_channels, kernel_size=1, bias=True)
        self._built = True
        self.to(device=device)

    def forward(self, cond_dict: dict) -> dict:
        if not isinstance(cond_dict, dict):
            raise TypeError(f"cond_dict must be a dict, got {type(cond_dict)}")
        tensors = []
        for key in ("rms_vel", "migrated_image", "horizon", "well_log"):
            value = cond_dict.get(key, None)
            if torch.is_tensor(value):
                tensors.append(self._to_2d70(value))
        if not tensors:
            raise ValueError("No valid condition tensors found in cond_dict.")

        x = torch.cat(tensors, dim=1)
        if not self._built:
            self._build(in_channels=x.shape[1], device=x.device)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = self.head(x)
        return {"s16": x}


def _test_naive_resnet_cond_encoder_cpu() -> None:
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
    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")



if  __name__ == "__main__":
    _test_naive_resnet_cond_encoder_cpu()