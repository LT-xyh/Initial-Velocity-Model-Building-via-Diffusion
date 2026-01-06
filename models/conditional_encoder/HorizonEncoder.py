import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 基础块 ----------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = ((k - 1) // 2) * d
            else:
                p = (((k[0] - 1) // 2) * d, ((k[1] - 1) // 2) * d)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, k=3, s=1, d=dilation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNAct(ch, ch, k=3, s=1, d=1, act=False)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.act(x + y)


class SEBlock(nn.Module):
    """通道注意力（Squeeze-Excitation）"""

    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class SpatialAttn(nn.Module):
    """空间注意力（CBAM风格）：聚焦细线位置"""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        m = torch.cat([torch.max(x, dim=1, keepdim=True).values,
                       torch.mean(x, dim=1, keepdim=True)], dim=1)
        w = torch.sigmoid(self.conv(m))
        return x * w


# ---------- 高斯模糊（软带宽） ----------
def make_gaussian_kernel(size=7, sigma=1.5, device="cpu"):
    ax = torch.arange(size, device=device) - (size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel  # (K,K)


class FixedGaussianBlur(nn.Module):
    """单通道高斯模糊（可微，权重固定）"""

    def __init__(self, k=7, sigma=1.5):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.register_buffer("weight", torch.zeros(1, 1, k, k), persistent=False)
        self._init = False

    def _maybe_init(self, device):
        if not self._init or self.weight.device != device:
            ker = make_gaussian_kernel(self.k, self.sigma, device=device)
            self.weight.data = ker.view(1, 1, self.k, self.k)
            self._init = True

    def forward(self, x):
        # x: (B,1,H,W)
        self._maybe_init(x.device)
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=self.k // 2)


# ---------- Horizon Encoder 主体 ----------

class HorizonEncoder(nn.Module):
    """
    构造坐标通道与软带宽(soft band)增强; 再用轻量通道+空间注意力强调界面区域
    输入:  (B, 1, 70, 70)   二值层位图 (界面=1, 其他=0)
    输出:  (B, C_cond, 16, 16) 空间条件嵌入（用于与 latent 对齐的注入）
    设计关键:
      * 内部自动构造 4 个输入通道：原始mask、软带宽、y/x 归一化坐标
      * 早期尽量不下采样，先用注意力增强细线，再渐进下采样到 16×16
    """

    def __init__(self, c1=32, c2=48, c3=64, c_cond=16,
                 se_ratio=8, dropout=0.0, gaussian_k=7, gaussian_sigma=1.5):
        super().__init__()
        self.blur = FixedGaussianBlur(k=gaussian_k, sigma=gaussian_sigma)

        # stem: 输入通道 = 1(mask) + 1(soft band) + 2(coord)
        self.stem = ConvBNAct(4, c1, k=3, s=1)
        self.res1 = ResBlock(c1, dilation=1, dropout=dropout)
        self.se1 = SEBlock(c1, r=se_ratio)
        self.satt1 = SpatialAttn(k=7)

        # 轻下采样到 35×35（尽量晚一些下采样，保护细线）
        self.down1 = ConvBNAct(c1, c2, k=3, s=2)  # 70->35
        self.res2 = ResBlock(c2, dilation=2, dropout=dropout)
        self.se2 = SEBlock(c2, r=se_ratio)
        self.satt2 = SpatialAttn(k=7)

        # 再下采样到 ~18×18
        self.down2 = ConvBNAct(c2, c3, k=3, s=2)  # 35->18
        self.res3 = ResBlock(c3, dilation=2, dropout=dropout)
        self.se3 = SEBlock(c3, r=se_ratio)
        self.satt3 = SpatialAttn(k=7)

        # 精准对齐到 16×16
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        # 投影到条件通道
        self.proj = nn.Conv2d(c3, c_cond, kernel_size=1, bias=True)

        # 轻量正则
        self.ln = nn.Identity()  # 如需更稳，可换成 nn.GroupNorm(8, c3) 再 proj

    def _build_aug_channels(self, x):
        """
        构造增强输入通道：
          - soft band: 高斯模糊后的“增粗”层位
          - coords: 归一化坐标 [y,x] ∈ [-1,1]
        """
        B, _, H, W = x.shape
        soft = self.blur(x)  # (B,1,H,W)

        # 归一化坐标，随 batch/device 自适应
        device = x.device
        yy = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        # 拼成 4 通道输入
        return torch.cat([x, soft, yy, xx], dim=1)

    def forward(self, x):
        """
        x: (B,1,70,70)  二值层位图
        """
        B, C, H, W = x.shape
        assert C == 1 and H == 70 and W == 70, "期望输入为 (B,1,70,70)"

        xin = self._build_aug_channels(x)  # (B,4,70,70)

        h = self.stem(xin)  # 70×70
        h = self.res1(h);
        h = self.se1(h);
        h = self.satt1(h)

        h = self.down1(h)  # 35×35
        h = self.res2(h);
        h = self.se2(h);
        h = self.satt2(h)

        h = self.down2(h)  # 18×18
        h = self.res3(h);
        h = self.se3(h);
        h = self.satt3(h)

        h = self.pool(h)  # -> 16×16
        h = self.ln(h)
        e_cond = self.proj(h)  # -> (B, C_cond, 16, 16)
        return e_cond


def test_encoder():
    model = HorizonEncoder()
    input_tensor = torch.randn(4, 1, 70, 70)
    output = model(input_tensor)
    print(output.shape)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Test passed!")


if __name__ == "__main__":
    test_encoder()
