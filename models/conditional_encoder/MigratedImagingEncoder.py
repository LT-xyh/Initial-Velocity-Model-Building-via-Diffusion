import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ 基础模块 ------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=(1,1), p=None, d=1, act=True):
        super().__init__()
        if p is None:
            # padding 自适应以保持 "same"
            p = (((k - 1) // 2) * d) if isinstance(k, int) else (((k[0] - 1) // 2) * d, ((k[1] - 1) // 2) * d)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, k=3, s=(1,1), d=dilation)
        self.conv2 = ConvBNAct(ch, ch, k=3, s=(1,1), d=1, act=False)
        self.act   = nn.GELU()
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return self.act(x + y)

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation：轻量全局上下文/通道注意力 """
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, kernel_size=1)
        self.fc2 = nn.Conv2d(ch // r, ch, kernel_size=1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

# ------ 叠后偏移成像 Encoder ------

class MigratedImagingEncoder(nn.Module):
    """
    充分利用二维纹理/边缘信息，用各向异性下采样（纵向更强，横向更慢）、空洞卷积扩展感受野，并用全局上下文（SE/GC）做轻量注意力；最后 AdaptiveAvgPool2d 精准对齐到 16×16。
    输入:  叠后偏移图像 (B, 1, 1000, 70)  ->  输出: 空间条件嵌入 (B, C_cond, 16, 16)
        设计要点：
      - 先做轻微平滑，再用各向异性 stride (2,1) 逐级下采样 1000 -> 500 -> 250 -> 125；
      - 宽度方向 70 -> 35 用一次 stride (1,2)；
      - 间插空洞卷积的 ResBlock 扩大感受野以捕捉层状与断裂；
      - SE 全局上下文增强长程依赖；
      - 最后自适应池化到 (16,16)，1x1 投影到 C_cond。
    """
    def __init__(self, c1=32, c2=64, c3=96, c_cond=64, se_ratio=8):
        """
        c1,c2,c3 : 各阶段通道数
        c_cond   : 融合前该分支输出通道（建议 32/64/128 之一；与其他模态在融合处 concat）
        """
        super().__init__()

        # 预平滑（降低混叠风险）
        self.pre = ConvBNAct(1, c1, k=3, s=(1,1))

        # Stage 1: H 1000->500, W 70->70
        self.down1 = ConvBNAct(c1, c1, k=3, s=(2,1))     # 各向异性下采样（主要压垂向）
        self.res1a = ResBlock(c1, dilation=1)
        self.res1b = ResBlock(c1, dilation=2)            # 空洞卷积扩感受野

        # Stage 2: H 500->250
        self.to_c2 = ConvBNAct(c1, c2, k=3, s=(1,1))
        self.down2 = ConvBNAct(c2, c2, k=3, s=(2,1))
        self.res2a = ResBlock(c2, dilation=1)
        self.res2b = ResBlock(c2, dilation=2)

        # Stage 3: H 250->125
        self.to_c3 = ConvBNAct(c2, c3, k=3, s=(1,1))
        self.down3 = ConvBNAct(c3, c3, k=3, s=(2,1))
        self.res3a = ResBlock(c3, dilation=1)
        self.res3b = ResBlock(c3, dilation=4)            # 更大感受野，利于跨层追踪

        # 宽度方向 70->35（一次性减半）
        self.down_w = ConvBNAct(c3, c3, k=(3,3), s=(1,2))

        # 轻量全局上下文
        self.se = SEBlock(c3, r=se_ratio)

        # 统一到 (16,16)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        # 投影到条件通道
        self.proj = nn.Conv2d(c3, c_cond, kernel_size=1, bias=True)

    def forward(self, x):
        """
        x: (B, 1, 1000, 70)
        """
        B, C, H, W = x.shape
        assert C == 1 and H == 1000 and W == 70, "期望输入为 (B,1,1000,70)"

        h = self.pre(x)            # (B, c1, 1000, 70)

        h = self.down1(h)          # (B, c1,  500, 70)
        h = self.res1a(h)
        h = self.res1b(h)

        h = self.to_c2(h)          # (B, c2,  500, 70)
        h = self.down2(h)          # (B, c2,  250, 70)
        h = self.res2a(h)
        h = self.res2b(h)

        h = self.to_c3(h)          # (B, c3,  250, 70)
        h = self.down3(h)          # (B, c3,  125, 70)
        h = self.res3a(h)
        h = self.res3b(h)

        h = self.down_w(h)         # (B, c3,  125, 35)

        h = self.se(h)             # 通道注意力全局增强

        h = self.pool(h)           # -> (B, c3, 16, 16)
        e_cond = self.proj(h)      # -> (B, c_cond, 16, 16)
        return e_cond






def test_migrated_imaging_encoder():
    model = MigratedImagingEncoder()
    input_tensor = torch.randn(4, 1, 1000, 70)
    output = model(input_tensor)
    print(output.shape)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Test passed!")


if __name__ == "__main__":
    test_migrated_imaging_encoder()