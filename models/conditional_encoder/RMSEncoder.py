import torch
import torch.nn as nn


class ConvBNGELU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RMSVelocityEncoder(nn.Module):
    """
    RMS速度条件编码器
    使用1D卷积捕捉单道的速度变化 + 2D卷积提取空间结构
    输入:  RMS 速度 (B, 1, 1000, 70)  —— [时间, 横向]
    输出:  空间条件嵌入 (B, C_cond, 16, 16)
    """

    def __init__(self, c_temporal=8, c_mid=64, c_cond=32):
        """
        c_temporal:  逐道时序编码后的通道数（把 1D 时序信息变成多个“谱/趋势”通道）
        c_mid:       2D 编码中间通道
        c_cond:      最终条件嵌入通道数（用于与其他模态融合前的 RMS 分支输出）
        """
        super().__init__()
        # ---- 1) 逐道时序编码 (Conv1d) ----
        # 对每条道 (T=1000) 做 1D 卷积提取纵向趋势，再自适应平均池化到 70
        self.temporal_conv1 = nn.Conv1d(1, c_temporal, kernel_size=7, padding=3, bias=False)
        self.temporal_bn1 = nn.BatchNorm1d(c_temporal)
        self.temporal_conv2 = nn.Conv1d(c_temporal, c_temporal, kernel_size=5, padding=2, bias=False)
        self.temporal_bn2 = nn.BatchNorm1d(c_temporal)
        self.temporal_act = nn.GELU()
        self.temporal_pool = nn.AdaptiveAvgPool1d(70)  # 1000 -> 70（防混叠/低通性质）

        # ---- 2) 2D 空间编码 (Conv2d) ----
        # 输入将是 (B, c_temporal, 70, 70)
        self.stem = ConvBNGELU(c_temporal, c_mid // 2, k=3, s=1, p=1)
        self.block1 = nn.Sequential(
            ConvBNGELU(c_mid // 2, c_mid, k=3, s=1, p=1),
            ConvBNGELU(c_mid, c_mid, k=3, s=1, p=1),
        )
        # 轻微下采样，控制感受野；也可以全部留给自适应池化
        self.down = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=2, padding=1)  # 70->35
        self.block2 = nn.Sequential(
            ConvBNGELU(c_mid, c_mid, k=3, s=1, p=1),
            ConvBNGELU(c_mid, c_mid, k=3, s=1, p=1),
        )

        # 统一对齐到 (16,16)
        self.spatial_pool = nn.AdaptiveAvgPool2d((16, 16))

        # 投影到目标条件通道
        self.proj = nn.Conv2d(c_mid, c_cond, kernel_size=1, bias=True)

    def forward(self, x):
        """
        x: (B, 1, 1000, 70)  [C=1, T=1000, X=70]
        """
        B, C, T, X = x.shape
        assert C == 1 and T == 1000, "期望输入为 (B,1,1000,70)"
        # ---- 1) 逐道时序编码（对每个横向位置独立编码）----
        # 变形为 (B*X, 1, T)
        x_1d = x.permute(0, 3, 1, 2).contiguous().view(B * X, 1, T)  # (B*70,1,1000)

        h = self.temporal_conv1(x_1d)
        h = self.temporal_act(self.temporal_bn1(h))
        h = self.temporal_conv2(h)
        h = self.temporal_act(self.temporal_bn2(h))  # (B*X, c_temporal, 1000)
        h = self.temporal_pool(h)  # -> (B*X, c_temporal, 70)

        # 还原为 2D: (B, X, c_temporal, 70) -> (B, c_temporal, 70, X)
        h = h.view(B, X, -1, 70).permute(0, 2, 3, 1).contiguous()  # (B, c_temporal, 70, 70)

        # ---- 2) 2D 空间编码 ----
        h = self.stem(h)  # (B, c_mid//2, 70, 70)
        h = self.block1(h)  # (B, c_mid,    70, 70)
        h = self.down(h)  # (B, c_mid,    35, 35)
        h = self.block2(h)  # (B, c_mid,    35, 35)

        # 统一到 (16,16)
        h = self.spatial_pool(h)  # (B, c_mid, 16, 16)

        # 投影为条件通道
        e_cond = self.proj(h)  # (B, c_cond, 16, 16)
        return e_cond


def test_rms_encoder():
    model = RMSVelocityEncoder()
    rms_vel = torch.randn(4, 1, 1000, 70)
    output = model(rms_vel)
    print(output.shape)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Test passed!")


if __name__ == "__main__":
    test_rms_encoder()
