import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- 通用小模块 ---------
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
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ColumnAttn(nn.Module):
    """ 列注意力：沿深度平均->按宽度(列)做深度可分离卷积获得列权重 """

    def __init__(self, ch, k=7):
        super().__init__()
        self.dw1d = nn.Conv2d(ch, ch, kernel_size=(1, k), padding=(0, k // 2), groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B,C,H,W)
        m = x.mean(dim=2, keepdim=True)  # (B,C,1,W)
        w = torch.sigmoid(self.pw(self.dw1d(m)))  # (B,C,1,W)
        return x * w  # 广播到H


# --------- 固定横向高斯：给竖线“增粗” ---------
def gaussian1d_kernel(k=7, sigma=1.5, device="cpu"):
    ax = torch.arange(k, device=device) - (k - 1) / 2.0
    ker = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    ker = ker / ker.sum()
    return ker  # (k,)


class FixedGaussianBlurWidth(nn.Module):
    """ 仅沿宽度方向做高斯模糊（1 x k），增强竖线的可见宽度 """

    def __init__(self, k=7, sigma=1.5):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.register_buffer("weight", torch.zeros(1, 1, 1, k), persistent=False)
        self._init = False

    def _maybe_init(self, device):
        if not self._init or self.weight.device != device:
            ker = gaussian1d_kernel(self.k, self.sigma, device=device).view(1, 1, 1, self.k)
            self.weight.data = ker
            self._init = True

    def forward(self, x):
        self._maybe_init(x.device)
        return F.conv2d(x, self.weight, stride=1, padding=(0, self.k // 2))


# --------- 竖向1D支路：逐列曲线编码 ---------
class Column1DEncoder(nn.Module):
    """ 对每个列的速度曲线(长度H)做1D卷积，得到 c_temporal 个“曲线模式”通道 """

    def __init__(self, c_temporal=8):
        super().__init__()
        self.conv1 = nn.Conv1d(1, c_temporal, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(c_temporal)
        self.conv2 = nn.Conv1d(c_temporal, c_temporal, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_temporal)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B,1,H,W) -> (B*W,1,H) -> (B,c_temporal,H,W)
        B, C, H, W = x.shape
        t = x.permute(0, 3, 1, 2).contiguous().view(B * W, 1, H)
        t = self.act(self.bn1(self.conv1(t)))
        t = self.act(self.bn2(self.conv2(t)))
        t = t.view(B, W, -1, H).permute(0, 2, 3, 1).contiguous()  # (B,c_temporal,H,W)
        return t


# --------- 主体：Well Log Encoder ---------
class WellLogEncoder(nn.Module):
    """
    输入:  (B,1,70,70)，仅有7条竖向井道为非零速度，其余为0
    输出:  (B, C_cond, 16,16)
    设计:
      * 构造 5 通道输入: 原始x / 有效性mask / 横向软带宽 / y坐标 / x坐标
      * 双分支: 竖向1D曲线编码 + 2D卷积上下文，列注意力突出含井列
      * 渐进下采样到 ~18×18，最后自适应到 16×16，1×1投影到 C_cond
    """

    def __init__(self, c_temporal=8, c2d_1=32, c2d_2=48, c2d_3=64, c_cond=16,
                 se_ratio=8, dropout=0.0, k_soft=7, sigma_soft=1.5):
        super().__init__()
        self.soft = FixedGaussianBlurWidth(k=k_soft, sigma=sigma_soft)
        self.col1d = Column1DEncoder(c_temporal=c_temporal)

        # 2D支路 stem（输入通道=5）
        self.stem = ConvBNAct(5, c2d_1, k=(5, 3))  # 竖向加大感受野
        self.res1 = ResBlock(c2d_1, dilation=1, dropout=dropout)
        self.se1 = SEBlock(c2d_1, r=se_ratio)
        self.catt1 = ColumnAttn(c2d_1, k=7)

        # 下采样 70->35
        self.down1 = ConvBNAct(c2d_1, c2d_2, k=3, s=2)
        self.res2 = ResBlock(c2d_2, dilation=2, dropout=dropout)
        self.se2 = SEBlock(c2d_2, r=se_ratio)
        self.catt2 = ColumnAttn(c2d_2, k=7)

        # 下采样 35->18（四舍五入）
        self.down2 = ConvBNAct(c2d_2, c2d_3, k=3, s=2)
        self.res3 = ResBlock(c2d_3, dilation=2, dropout=dropout)
        self.se3 = SEBlock(c2d_3, r=se_ratio)
        self.catt3 = ColumnAttn(c2d_3, k=7)

        # 双分支融合（concat 后用 1×1 整合）
        self.fuse = ConvBNAct(c2d_3 + c_temporal, c2d_3, k=1, s=1)

        # 对齐到 16×16并投影
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.proj = nn.Conv2d(c2d_3, c_cond, kernel_size=1, bias=True)

    def _build_inputs(self, x):
        """
        构造增强输入:
          - mask: (x!=0) 的有效性
          - soft_w: 仅横向高斯模糊，给竖线“增粗”以抗下采样
          - coords: 归一化 y/x 坐标
        """
        B, _, H, W = x.shape
        device = x.device
        mask = (x.abs() > 0).float()
        soft_w = self.soft(x)

        yy = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        return torch.cat([x, mask, soft_w, yy, xx], dim=1)  # (B,5,H,W)

    def forward(self, x):
        """
        x: (B,1,70,70)
        """
        B, C, H, W = x.shape
        assert C == 1 and H == 70 and W == 70, "期望输入为 (B,1,70,70)"

        # 竖向1D支路（逐列曲线编码）
        t_feat = self.col1d(x)  # (B,c_temporal,70,70)

        # 2D支路（上下文 + 列注意力）
        xin = self._build_inputs(x)  # (B,5,70,70)
        h = self.stem(xin)
        h = self.res1(h);
        h = self.se1(h);
        h = self.catt1(h)

        h = self.down1(h)  # -> 35×35
        h = self.res2(h);
        h = self.se2(h);
        h = self.catt2(h)

        h = self.down2(h)  # -> 18×18
        h = self.res3(h);
        h = self.se3(h);
        h = self.catt3(h)

        # 双分支融合（将1D特征按空间对齐后拼接）
        # t_feat 仍是 70×70，需要下采样到 h 的大小(≈18×18) 再融合
        t_ds = F.adaptive_avg_pool2d(t_feat, h.shape[-2:])  # -> (B,c_temporal,18,18)
        h = torch.cat([h, t_ds], dim=1)
        h = self.fuse(h)

        # 对齐 latent 尺度并投影
        h = self.pool(h)  # -> (B, c2d_3, 16,16)
        e_cond = self.proj(h)  # -> (B, C_cond, 16,16)
        return e_cond


def test_encoder():
    model = WellLogEncoder()
    input_tensor = torch.randn(4, 1, 70, 70)
    output = model(input_tensor)
    print(output.shape)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Test passed!")


if __name__ == "__main__":
    test_encoder()
