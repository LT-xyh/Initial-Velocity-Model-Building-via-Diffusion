import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1) UNet2DConditionModel 的封装：forward(z_t, t, cond_embed)
#    - 支持 cond_embed 为 Tensor: (B,64,16,16)  -> cross-attn tokens
#    - 或 cond_embed 为 dict: {'s16','s32','s64','s70'} -> cross-attn(token) + 额外残差(Adapter)
# ------------------------------------------------------------
try:
    from diffusers.models import UNet2DConditionModel
except ImportError as e:
    raise ImportError("pip install diffusers>=0.25.0") from e


class _PyramidAdapter(nn.Module):
    """
    生成与 UNet down blocks **输出**对齐的额外残差，顺序与 down blocks 一一对应。
    target_out_sizes 例：当 sample_size=16 且 4 个 down blocks（前3个下采样） -> [8,4,2,2]
    """
    def __init__(self,
                 in_dict_channels: dict[str, int],
                 level_channels: tuple[int, ...],
                 target_out_sizes: list[int]):
        super().__init__()
        assert len(level_channels) == len(target_out_sizes), "层数与目标尺寸不一致"
        self.in_dict_channels = in_dict_channels
        self.level_channels = level_channels
        self.target_out_sizes = target_out_sizes  # 与 down blocks 对齐

        # 可用的条件尺度名及其分辨率（按从大到小）
        self.scales = [('s70', 70), ('s64', 64), ('s32', 32), ('s16', 16)]

        projs = []
        for ch_out, _ in zip(level_channels, target_out_sizes):
            projs.append(nn.Sequential(
                nn.Conv2d( max(in_dict_channels.values()), ch_out, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out),
                nn.SiLU(),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            ))
        self.level_projs = nn.ModuleList(projs)

        mid_ch = level_channels[-1]
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, mid_ch), num_channels=mid_ch),
            nn.SiLU()
        )

    def _pick_src(self, cond_pyr: dict[str, torch.Tensor], target_sz: int) -> torch.Tensor:
        # 选一个与 target_sz 最接近且存在的尺度
        avail = [(k, sz) for (k, sz) in self.scales if k in cond_pyr and cond_pyr[k] is not None]
        if not avail:
            raise ValueError("Adapter: no available cond feature maps in cond_pyr.")
        k_best = min(avail, key=lambda kv: abs(kv[1] - target_sz))[0]
        return cond_pyr[k_best]

    def forward(self, cond_pyr: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype):
        cond_pyr = {k: (v.to(device=device, dtype=dtype) if v is not None else None)
                    for k, v in cond_pyr.items()}

        down_residuals = []
        for i, (ch_out, sz) in enumerate(zip(self.level_channels, self.target_out_sizes)):
            x = self._pick_src(cond_pyr, sz)
            if x.shape[-2:] != (sz, sz):
                x = F.adaptive_avg_pool2d(x, (sz, sz))
            # 若输入通道与预设不符，先用 1x1 提到最大输入通道再投影（简单稳妥）
            if x.shape[1] != max(self.in_dict_channels.values()):
                x = F.conv2d(x, weight=torch.zeros(
                    max(self.in_dict_channels.values()), x.shape[1], 1, 1, device=x.device, dtype=x.dtype))
            proj = self.level_projs[i]
            down_residuals.append(proj(x))

        mid_residual = self.mid_proj(down_residuals[-1])
        return down_residuals, mid_residual




class ConditionedUNet2DWrapper(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        latent_size: int = 16,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512),
        attention_head_dim: int = 8,
        use_adapter: bool = True,
        check_shapes: bool = True,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.check_shapes = check_shapes

        self.unet = UNet2DConditionModel(
            sample_size=latent_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            down_block_types=("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"),
            up_block_types=("UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
            cross_attention_dim=64,
            attention_head_dim=attention_head_dim,
        )

        self.use_adapter = use_adapter
        if use_adapter:
            # === 关键：按“块输出”推导尺寸（除最后一块，其余块都会下采样）===
            n_levels = len(block_out_channels)
            target_out_sizes = []  # 每个 down block 的 **输出** 分辨率
            cur = latent_size
            for i in range(n_levels):
                if i < n_levels - 1:
                    cur = max(1, cur // 2)  # 有下采样
                else:
                    cur = cur              # 最后一块不再下降
                target_out_sizes.append(cur)
            # 对于 4 层 & sample_size=16 -> [8,4,2,2]

            in_dict_channels = {'s16': 64, 's32': 64, 's64': 64, 's70': 32}
            self.adapter = _PyramidAdapter(
                in_dict_channels=in_dict_channels,
                level_channels=block_out_channels,
                target_out_sizes=target_out_sizes
            )


    @staticmethod
    def _cond_to_tokens(cond_s16: torch.Tensor, pool: int | None = None):
        if pool is not None and pool > 1:
            cond_s16 = F.avg_pool2d(cond_s16, kernel_size=pool, stride=pool)
        b, c, h, w = cond_s16.shape
        return cond_s16.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def forward(self, z_t: torch.Tensor, t: torch.LongTensor, cond_embed):
        device = z_t.device
        dtype = self.unet.dtype

        if isinstance(cond_embed, dict):
            assert 's16' in cond_embed and cond_embed['s16'] is not None, "cond_embed dict 必须包含 s16"
            cond_tokens = self._cond_to_tokens(cond_embed['s16'].to(device=device, dtype=dtype))
            if self.use_adapter:
                down_residuals, mid_residual = self.adapter(cond_embed, device, dtype)
                # --- 形状防御：确保第一层是 16x16，其余依次减半 ---
                if self.check_shapes:
                    expected_sizes = [self.latent_size // (2 ** i) for i in range(len(down_residuals))]
                    for i, (res, sz) in enumerate(zip(down_residuals, expected_sizes)):
                        if res.shape[-1] != sz or res.shape[-2] != sz:
                            down_residuals[i] = F.adaptive_avg_pool2d(res, (sz, sz))
            else:
                down_residuals, mid_residual = None, None

            out = self.unet(
                sample=z_t.to(device=device, dtype=dtype),
                timestep=t.to(device=device),
                encoder_hidden_states=cond_tokens,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual,
                return_dict=True
            ).sample
            return out

        else:
            cond_tokens = self._cond_to_tokens(cond_embed.to(device=device, dtype=dtype))
            out = self.unet(
                sample=z_t.to(device=device, dtype=dtype),
                timestep=t.to(device=device),
                encoder_hidden_states=cond_tokens,
                return_dict=True
            ).sample
            return out



# ------------------------------------------------------------
# 2) 完整扩散过程：训练前向 + 采样（DDIM/Prob-Flow）
#    - 可单独使用本类的 sampling；也可把本类的 model 交给你已有的 ProbabilityFlowODEScheduler.p_sample_loop
# ------------------------------------------------------------
class LatentDiffusionProcess(nn.Module):
    """
    端到端的扩散过程：
      - training_step(x0, cond_embed, loss_type='mse'/'p2')
      - sample(cond_embed, num_inference_steps=50, eta=0.0)  # DDIM/Prob-Flow
      - get_model(): 返回适配过的 model，可直接给你的 ProbabilityFlowODEScheduler.p_sample_loop 使用
    注：不调用外部条件编码器，cond_embed 由你在外部构造后传入。
    """
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "linear",
                 unet_kwargs: dict | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.timesteps = timesteps
        unet_kwargs = unet_kwargs or {}
        self.model = ConditionedUNet2DWrapper(**unet_kwargs)

        # ——调度表（与 DDPM 一致）——
        if beta_schedule == "linear":
            betas = self._linear_beta_schedule(timesteps)
        elif beta_schedule == "sigmoid":
            betas = self._sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError("beta_schedule must be 'linear' or 'sigmoid'")

        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.to(dtype))
        self.register_buffer("alphas", alphas.to(dtype))
        self.register_buffer("alphas_cumprod", a_bar.to(dtype))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(a_bar).to(dtype))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - a_bar).to(dtype))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / a_bar).to(dtype))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / a_bar - 1).to(dtype))

    # ---- schedules ----
    @staticmethod
    def _linear_beta_schedule(timesteps: int):
        scale = 1000 / timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 2e-2
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    @staticmethod
    def _sigmoid_beta_schedule(timesteps: int, start=-3, end=3, tau=1.0):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        ac = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return torch.clamp(betas, 0, 0.999).to(torch.float32)

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
        out = a.gather(0, t.clamp(max=a.shape[0]-1))
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    # ---- q(xt|x0) & x0(xt,eps) ----
    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor | None = None):
        noise = noise if noise is not None else torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        ).to(x0.dtype)

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.LongTensor, eps: torch.Tensor):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * eps
        ).to(xt.dtype)

    # ---- 训练前向 ----
    def training_step(self, x0: torch.Tensor, cond_embed, t: torch.LongTensor | None = None,
                      loss_type: str = "mse", p2_k: float = 1.0, p2_gamma: float = 0.5):
        device = x0.device
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)                              # q(xt|x0)
        eps_pred = self.model(xt, t, cond_embed)                      # 预测噪声

        if loss_type == "mse":
            loss = F.mse_loss(eps_pred, noise)
        elif loss_type == "p2":
            alpha_bar_t = self._extract(self.alphas_cumprod, t, x0.shape)
            snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
            w = (snr ** p2_gamma) / (p2_k + (snr ** p2_gamma))
            loss = (w * (eps_pred - noise) ** 2).mean()
        else:
            raise ValueError("loss_type must be 'mse' or 'p2'")
        return loss, {"eps_pred": eps_pred, "t": t}

    # ---- DDIM / Probability-Flow ODE 风格采样（eta=0 确定性）----
    def _make_inference_steps(self, num_inference_steps: int):
        step = max(self.timesteps // num_inference_steps, 1)
        steps = list(range(0, self.timesteps, step))
        if steps[-1] != self.timesteps - 1:
            steps[-1] = self.timesteps - 1
        return steps

    @torch.no_grad()
    def sample(self, cond_embed, num_inference_steps: int = 50, eta: float = 0.0,
               z_T: torch.Tensor | None = None, return_traj: bool = False):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        steps = self._make_inference_steps(num_inference_steps)
        B = cond_embed['s16'].shape[0] if isinstance(cond_embed, dict) else cond_embed.shape[0]

        xt = torch.randn(B, 16, 16, 16, device=device, dtype=dtype) if z_T is None else z_T.to(device, dtype)
        traj = [xt] if return_traj else None

        for i in range(len(steps) - 1, 0, -1):
            t = steps[i]
            s = steps[i - 1]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            eps = self.model(xt, t_batch, cond_embed)
            x0 = self.predict_x0_from_eps(xt, t_batch, eps)

            a_t = self.alphas_cumprod[t]
            a_s = self.alphas_cumprod[s]
            sigma_t = eta * torch.sqrt((1 - a_s) / (1 - a_t)) * torch.sqrt(torch.clamp(1 - a_t / a_s, min=0.0))
            xt = torch.sqrt(a_s) * x0 + torch.sqrt(torch.clamp(1 - a_s - sigma_t**2, min=0.0)) * eps
            if eta > 0:
                xt = xt + sigma_t * torch.randn_like(xt)

            if return_traj:
                traj.append(xt)

        # 跳到 s=0
        s = 0
        a_s = self.alphas_cumprod[s]
        t0_batch = torch.full((B,), steps[1], device=device, dtype=torch.long)
        eps = self.model(xt, t0_batch, cond_embed)
        x0 = self.predict_x0_from_eps(xt, t0_batch, eps)
        x0_hat = torch.sqrt(a_s) * x0 + torch.sqrt(1 - a_s) * eps

        if return_traj:
            traj.append(x0_hat)
            return traj, x0_hat
        return x0_hat

    # 把内部 model 暴露给你的 ProbabilityFlowODEScheduler 使用
    def get_model(self):
        return self.model


# ---------------------------- 用法示例 ----------------------------
if __name__ == "__main__":
    B = 2
    x0 = torch.randn(B, 16, 16, 16)               # 训练用的目标 latent
    cond = torch.randn(B, 64, 16, 16)             # 或者:
    cond_pyr = {'s16': cond, 's32': torch.randn(B,64,32,32), 's64': torch.randn(B,64,64,64), 's70': torch.randn(B,32,70,70)}

    ldp = LatentDiffusionProcess(
        timesteps=1000,
        beta_schedule="linear",
        unet_kwargs=dict(latent_channels=16, latent_size=16, use_adapter=True),
    )
    ldp = ldp.to(x0.device)

    # 训练前向
    loss, aux = ldp.training_step(x0, cond)        # 传 Tensor
    print("train loss (tensor cond):", loss.item())

    loss2, _ = ldp.training_step(x0, cond_pyr)     # 传 dict（会走 adapter 残差）
    print("train loss (pyr cond):", loss2.item())

    # 采样（确定性轨迹，eta=0）
    x0_hat = ldp.sample(cond_pyr, num_inference_steps=50, eta=0.0)
    print("sampled:", x0_hat.shape)

    # ——如果要用你现成的 ProbabilityFlowODEScheduler：
    # scheduler = ProbabilityFlowODEScheduler(timesteps=1000, ddim_step=100)
    # model = ldp.get_model()    # 符合接口的 model(z_t, t, cond_embed)
    # z_T = torch.randn(B, 16, 16, 16)
    # z0_hat = scheduler.p_sample_loop(z_T, model, cond_pyr)
