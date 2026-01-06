import torch
import torch.nn as nn


import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_utils import SchedulerOutput


class ProbabilityFlowODEScheduler(nn.Module):
    """
    DDIM / Probability-Flow (eta=0) style scheduler, with a diffusers-like API.

    - 训练前向:
        add_noise(x0, noise, timesteps)  # 等价 q(x_t | x_0)
        q_sample(...) 是 add_noise 的别名

    - 推理反向 (确定性):
        set_timesteps(num_inference_steps)
        for t in scheduler.timesteps:          # 通常是降序
            eps = model(...)
            out = scheduler.step(eps, t, x_t)
            x_t = out.prev_sample

    Attributes
    ----------
    num_train_timesteps : int
        训练时的总时间步（离散步数）。
    num_inference_steps : int | None
        推理时使用的步数（可小于 num_train_timesteps）。

    timesteps : torch.LongTensor
        推理时使用的离散时间步列表（降序），与 diffusers 一致。
    """

    def __init__(self, num_train_timesteps: int = 1000,
                 ddim_step: int | None = 100,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.config.num_train_timesteps = self.num_train_timesteps

        # ---- beta / alpha ----
        betas = self.linear_beta_schedule(self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 注册 buffer，方便 .to(device, dtype)
        self.register_buffer("betas", betas.to(dtype))
        self.register_buffer("alphas", alphas.to(dtype))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(dtype))
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.sqrt(alphas_cumprod).to(dtype))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).to(dtype),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod).to(dtype),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0).to(dtype),
        )

        # 推理步数
        self.num_inference_steps: int | None = None

        # 初始化 timesteps（diffusers 风格：降序）
        if ddim_step is None:
            self.set_timesteps(num_inference_steps=50)
        else:
            step = max(1, int(ddim_step))
            steps = list(range(0, self.num_train_timesteps, step))
            if steps[-1] != self.num_train_timesteps - 1:
                steps[-1] = self.num_train_timesteps - 1
            # 降序存储
            self.timesteps = torch.tensor(steps[::-1], dtype=torch.long)

    # --------- 与旧代码兼容的小别名 ---------
    def set_inference_steps(self, num_inference_steps: int = 50, device=None):
        """
        兼容你之前的命名，内部直接调用 set_timesteps。
        """
        return self.set_timesteps(num_inference_steps, device=device)

    @property
    def time_steps(self):
        """
        兼容你原来的 self.time_steps 用法，直接返回 self.timesteps。
        （注意：这里是降序，与 diffusers 一致）
        """
        return self.timesteps

    # --------- 时间网格：diffusers 风格 ---------
    @torch.no_grad()
    def set_timesteps(self, num_inference_steps: int = 50, device=None):
        """
        设置推理使用的离散时间步列表 self.timesteps（降序），
        对齐 diffusers.SchedulerMixin 的接口。
        """
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")

        self.num_inference_steps = int(num_inference_steps)
        step = max(1, self.num_train_timesteps // self.num_inference_steps)

        steps = list(range(0, self.num_train_timesteps, step))
        if steps[-1] != self.num_train_timesteps - 1:
            steps[-1] = self.num_train_timesteps - 1

        timesteps = torch.tensor(steps[::-1], dtype=torch.long)  # 降序
        if device is not None:
            timesteps = timesteps.to(device)
        self.timesteps = timesteps

    # --------- 工具函数 ---------
    @staticmethod
    def linear_beta_schedule(timesteps=1000):
        """
        与 DDPM 论文相同的线性 beta 日程。
        """
        scale = 1000 / timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 2e-2
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    @staticmethod
    def sigmoid_beta_schedule(timesteps=1000, start=-3, end=3, tau=1.0):
        """
        备用 sigmoid 日程。
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
        v_start = torch.tensor(start / tau, dtype=torch.float32).sigmoid()
        v_end = torch.tensor(end / tau, dtype=torch.float32).sigmoid()
        ac = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return torch.clamp(betas, 0, 0.999).to(torch.float32)

    @staticmethod
    def _as_batch_times(t, B, device):
        """
        把 int / 0-dim tensor / shape=(B,) tensor 统一成 (B,) 的 long tensor。
        """
        if isinstance(t, int):
            return torch.full((B,), t, dtype=torch.long, device=device)
        if isinstance(t, torch.Tensor):
            if t.ndim == 0:
                return t.view(1).repeat(B).to(torch.long).to(device)
            return t.to(torch.long).to(device)
        raise TypeError("t must be int or Tensor")

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        从 1D tensor a[0..T-1] 中按 batch index t[b] 取值，并 reshape 为可广播形状。
        """
        a = a.to(t.device)
        t = t.clamp(0, a.shape[0] - 1)
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    # --------- diffusers: scale_model_input ---------
    def scale_model_input(self, sample: torch.FloatTensor,
                          timestep: int | torch.Tensor | None = None) -> torch.FloatTensor:
        """
        对齐 diffusers 接口；这里不需要缩放，直接返回即可。
        """
        return sample

    # --------- 前向 q(x_t | x_0)：diffusers 的 add_noise + 你的 q_sample ---------
    def add_noise(self,
                  original_samples: torch.FloatTensor,
                  noise: torch.FloatTensor,
                  timesteps: torch.Tensor | int) -> torch.FloatTensor:
        """
        diffusers 风格前向加噪:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        B = original_samples.shape[0]
        if noise is None:
            noise = torch.randn_like(original_samples)

        t = self._as_batch_times(timesteps, B, original_samples.device)

        return (
            self.extract(self.sqrt_alphas_cumprod, t, original_samples.shape) * original_samples +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, original_samples.shape) * noise
        ).to(original_samples.dtype)

    def q_sample(self, x_start, t, noise=None):
        """
        兼容你原来的命名；内部调用 add_noise。
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.add_noise(x_start, noise, t)

    def predict_start_from_noise(self, x_t, t, noise):
        """
        x0 = 1/sqrt(alpha_bar_t) * x_t - sqrt(1/alpha_bar_t - 1) * eps
        """
        B = x_t.shape[0]
        t = self._as_batch_times(t, B, x_t.device)
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        ).to(x_t.dtype)

    # --------- 反向一步：diffusers 风格 step(...) ---------
    @torch.no_grad()
    def step(self,
             model_output: torch.FloatTensor,    # eps_t
             timestep: int | torch.Tensor,       # 当前 t（来自 self.timesteps 的一个元素）
             sample: torch.FloatTensor,          # x_t
             return_dict: bool = True) -> SchedulerOutput | tuple:
        """
        执行一次确定性 DDIM / 概率流 ODE 步进：
            t -> prev_t  （prev_t 从 self.timesteps 中自动推出来）

        - model_output: UNet 输出的噪声 eps_t
        - timestep: 当前离散时间步（int 或 0-dim Tensor）
        - sample: 当前 x_t

        返回:
        - SchedulerOutput(prev_sample= x_{prev_t})
        """
        B = sample.shape[0]
        device = sample.device

        # t 批量
        t_batch = self._as_batch_times(timestep, B, device)

        # 找到 prev_t：在 self.timesteps（降序）里找 index，下一个就是 prev_t
        if isinstance(timestep, torch.Tensor):
            t_int = int(timestep.detach().item())
        else:
            t_int = int(timestep)

        timesteps = self.timesteps.to(device)
        # 允许训练/推理 timesteps 不完全一致时的 fallback
        idxs = (timesteps == t_int).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            # 如果 timestep 不在 timesteps 里，就假设 prev_t = max(t - 1, 0)
            prev_t_int = max(t_int - 1, 0)
        else:
            idx = idxs[0].item()
            if idx == timesteps.shape[0] - 1:
                prev_t_int = 0
            else:
                prev_t_int = int(timesteps[idx + 1].item())

        s_batch = self._as_batch_times(prev_t_int, B, device)

        # 1) x0_t
        x0_t = self.predict_start_from_noise(sample, t_batch, model_output)

        # 2) DDIM(eta=0) / 概率流 ODE 形式：
        #    x_{prev} = sqrt(alpha_bar_prev) * x0_t + sqrt(1 - alpha_bar_prev) * eps_t
        a_s = self.extract(self.alphas_cumprod, s_batch, sample.shape)  # alpha_bar_{prev}
        prev_sample = torch.sqrt(a_s) * x0_t + torch.sqrt(1.0 - a_s) * model_output
        prev_sample = prev_sample.to(sample.dtype)

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    # --------- 可选：一个简单的整体采样 loop（方便你继续用） ---------
    @torch.no_grad()
    def p_sample_loop(self, model, cond_embed, shape, num_inference_steps: int | None = None,
                      device: torch.device | None = None):
        """
        方便从这个 scheduler 直接做完整采样：
            - model: UNet, 调用方式 model(x_t, t_batch, cond_embed) -> eps_t
            - cond_embed: 条件特征
            - shape: 初始噪声的形状 (B, C, H, W)

        这个接口是“你当前写法”的升级版，但内部已经用 diffusers 风格 step。
        """
        if device is None:
            device = cond_embed.device

        if num_inference_steps is not None:
            self.set_timesteps(num_inference_steps, device=device)

        timesteps = self.timesteps.to(device)
        x_t = torch.randn(shape, device=device)

        for t in timesteps:
            t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
            model_in = self.scale_model_input(x_t, t)
            eps_t = model(model_in, t_batch, cond_embed)
            out = self.step(eps_t, t, x_t, return_dict=True)
            x_t = out.prev_sample

        return x_t




if __name__ == '__main__':
    def test_scheduler():
        scheduler = ProbabilityFlowODEScheduler(num_train_timesteps=1000, ddim_step=100, dtype=torch.float32)
        print(scheduler.time_steps)
        print(scheduler.time_steps[torch.randint(0, scheduler.time_steps.numel(), (2,))])
    test_scheduler()