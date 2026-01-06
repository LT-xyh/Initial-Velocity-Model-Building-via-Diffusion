import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from diffusers.models.resnet import ResnetBlock2D
from utils.modules import ResBlock


# ---- 小工具 ----
def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    # scores, mask shape: (B, M, 1, 1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    # 处理全0的极端情况（避免 -inf nan）
    all_zero = (mask.sum(dim=dim, keepdim=True) == 0)
    scores = scores.clone()
    scores[all_zero.expand_as(scores)] = 0.0
    w = torch.softmax(scores, dim=dim)
    # 对全0情况，平均分配
    if all_zero.any():
        w = torch.where(all_zero.expand_as(w), torch.full_like(w, 1.0 / w.size(dim)), w)
    return w

class ConvBNAct(nn.Module):
    """
    卷积 + 批归一化 + 激活函数模块
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = ((k - 1) // 2) * d
            else:
                p = (((k[0]-1)//2)*d, ((k[1]-1)//2)*d)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---- CBAM 注意力（通道 + 空间） ----
class CBAM(nn.Module):
    def __init__(self, ch, r=8, k_spatial=7):
        super().__init__()
        mid = max(1, ch // r)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1), nn.GELU(),
            nn.Conv2d(mid, ch, 1)
        )
        self.spatial = nn.Conv2d(2, 1, k_spatial, padding=k_spatial//2, bias=False)

    def forward(self, x):
        # channel attention
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        w_c = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x   = x * w_c
        # spatial attention
        m = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], dim=1)
        w_s = torch.sigmoid(self.spatial(m))
        return x * w_s

# ---- 模态权重打分器（全局） ----
class ModalityScorer(nn.Module):
    """
    将每个模态特征 F_i -> 标量分数 s_i (B,1,1,1)，用于 softmax 门控
    """
    def __init__(self, ch_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, hidden, kernel_size=1), nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1)
        )
    def forward(self, x):
        # 先做 GAP 再打分：减少空间噪声
        g = F.adaptive_avg_pool2d(x, 1)
        return self.net(g)  # (B,1,1,1)

# ---- 多模态融合模块 ----
class MultiModalFusion(nn.Module):
    """
    将多分支条件嵌入融合为统一 E_cond
    Inputs: dict(name -> feat)；feat shape 均为 (B, C_i, 16, 16)
            缺失模态可以不提供或传 None
    Output:
        cond_map: (B, C_out, 16, 16)
        cond_vec: (B, D)  可选（给 AdaIN/FiLM）
    """
    def __init__(self,
                 in_channels: Dict[str, int],        # e.g. {'rms':32,'migr':64,'horizon':16,'well':16}
                 C_out: int = 64,                    # 融合后输出通道
                 C_mid: int = 128,                   # 融合中间通道
                 score_hidden: int = 32,             # 门控打分器隐层
                 use_cbam: bool = True,
                 return_vector_dim: Optional[int] = 0,  # >0 时返回 cond_vec
                 modality_dropout_p: float = 0.0      # 训练时随机丢弃模态的概率（每模态独立）
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.names = list(in_channels.keys())
        self.modality_dropout_p = modality_dropout_p

        # 每个模态一个 scorer
        self.scorers = nn.ModuleDict({
            name: ModalityScorer(ch_in=ch, hidden=score_hidden)
            for name, ch in in_channels.items()
        })

        # 融合前先 1x1 统一每个模态通道到一个较小维度（可选：这里直接拼接也可以）
        self.unify = nn.ModuleDict({
            name: nn.Identity()  # 如需先对齐通道，这里可换成 Conv1x1(ch_in -> ch_in)
            for name in in_channels.keys()
        })

        # concat 后降维 + 混合
        self.fuse_in = sum(in_channels.values())
        self.redu = ResBlock(self.fuse_in, C_mid, stride=1)# ConvBNAct(self.fuse_in, C_mid, k=1, s=1)

        # self.mix1 = ConvBNAct(C_mid, C_mid, k=3, s=1)
        self.mix1 = ResBlock(C_mid, C_mid, stride=1)
        # self.mix2 = ConvBNAct(C_mid, C_mid, k=3, s=1)
        self.mix2 = ResBlock(C_mid, C_mid, stride=1)

        self.cbam = CBAM(C_mid) if use_cbam else nn.Identity()

        # 残差输出
        self.out_proj = nn.Conv2d(C_mid, C_out, kernel_size=1, bias=True)

        # 全局向量分支（可选）
        if return_vector_dim and return_vector_dim > 0:
            self.vec_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(C_mid, return_vector_dim)
            )
        else:
            self.vec_head = None

    def _maybe_dropout(self, feats: Dict[str, Optional[torch.Tensor]]):
        """
        训练时按概率丢弃某些模态（Classifier-Free 风格），以提升鲁棒性
        """
        if not self.training or self.modality_dropout_p <= 0:
            return feats
        out = {}
        for name, f in feats.items():
            if f is None:
                out[name] = None
            else:
                drop = torch.rand((), device=f.device) < self.modality_dropout_p
                out[name] = None if drop else f
        # 防止全丢：若全为 None，随机恢复一个
        if all(v is None for v in out.values()):
            # 随机挑一个有值的原模态恢复
            avail = [k for k,v in feats.items() if v is not None]
            if len(avail) > 0:
                pick = avail[torch.randint(len(avail), (1,)).item()]
                out[pick] = feats[pick]
        return out

    def forward(self, feats: Dict[str, Optional[torch.Tensor]]):
        """
        feats: {'rms': (B,32,16,16), 'migr': (B,64,16,16), 'horizon': (B,16,16,16), 'well': (B,16,16,16)}
               缺失/不使用的模态可不传或传 None
        """
        feats = self._maybe_dropout(feats)

        # 仅保留存在的模态
        present = [(n, f) for n, f in feats.items() if (f is not None)]
        if len(present) == 0:
            raise ValueError("No modalities provided to MultiModalFusion.")

        xs = []
        scores = []
        masks = []
        for name, f in present:
            # 可在此插入 self.unify[name](f) 做通道对齐
            u = self.unify[name](f)                # (B,C_i,16,16)
            xs.append(u)
            s = self.scorers[name](u)              # (B,1,1,1)
            scores.append(s)
            masks.append(torch.ones_like(s))       # mask=1（存在）
        # 对缺失模态构造 0 特征 + 0 mask（用于 softmax 规范化）
        missing = [(n, None) for n in self.names if feats.get(n, None) is None]
        for name, _ in missing:
            B = xs[0].size(0)
            device = xs[0].device
            s = torch.zeros(B, 1, 1, 1, device=device)
            scores.append(s)
            masks.append(torch.zeros_like(s))      # mask=0（缺失）
            # 不把缺失模态加入 xs（不参与 concat），只在 softmax 权重里占位

        # 计算模态 softmax 权重
        scores = torch.stack(scores, dim=1)    # (B, M, 1, 1)  按 present+missing 的顺序
        masks  = torch.stack(masks,  dim=1)    # (B, M, 1, 1)
        weights = masked_softmax(scores, masks, dim=1)  # (B, M, 1, 1)

        # 将权重仅应用到“存在的模态”特征上（保持顺序一致）
        # weights 的前 len(present) 个对应 xs
        gated_xs = []
        for i, (_, f) in enumerate(present):
            w = weights[:, i, ...]             # (B,1,1,1)
            gated_xs.append(xs[i] * w)         # broadcast 到通道/空间

        # concat → 降维融合
        x = torch.cat(gated_xs, dim=1)         # (B, sumC_present, 16,16)
        # print(f'x: {x.shape}')
        x = self.redu(x)                       # (B, C_mid, 16,16)
        # 轻 residual mixing
        y = self.mix1(x)
        y = self.mix2(y)
        x = x + y
        x = self.cbam(x)

        cond_map = self.out_proj(x)            # (B, C_out, 16,16)

        if self.vec_head is not None:
            cond_vec = self.vec_head(x)        # (B, D)
            return {'map': cond_map, 'vec': cond_vec, 'weights': weights}
        else:
            return {'map': cond_map, 'weights': weights}


def test_fusion():
    e_rms = torch.sigmoid(torch.randn(4, 32, 16, 16))  # (B,32,16,16)
    e_mig = torch.sigmoid(torch.randn(4, 64, 16, 16))  # (B,64,16,16)
    e_hor = torch.sigmoid(torch.randn(4, 16, 16, 16))  # (B,16,16,16)
    e_wel = torch.sigmoid(torch.randn(4, 16, 16, 16))  # (B,16,16,16)

    model = MultiModalFusion(
        in_channels={'rms':32, 'migr':64, 'horizon':16, 'well':16},
        C_out=64,              # 统一条件嵌入通道，推荐 64
        C_mid=128,
        return_vector_dim=128, # 如需 AdaIN/FiLM 的全局向量；不需要可设 0
        modality_dropout_p=0 # 训练期可开；推理期会自动关闭
    )

    out = model({'rms': e_rms, 'migr': e_mig, 'horizon': e_hor, 'well': e_wel})
    E_cond = out['map']  # (B,64,16,16)   —— 供 CondAE 解码与扩散 U-Net 使用
    w_mod = out['weights']  # (B, M, 1,1)    —— 查看各模态门控权重（可做可解释性分析）
    cond_vec = out.get('vec')  # (B,128)        —— 若 return_vector_dim>0
    print(E_cond.shape, w_mod.shape, cond_vec.shape)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Test passed!")

if __name__ == "__main__":
    test_fusion()