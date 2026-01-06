import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    def __init__(self, timesteps=1000):
        """
        Diffusion Scheduler for forward diffusion process (q(x_t | x_0))
        :param timesteps: Number of diffusion steps
        :param beta_start: Starting value of beta (noise level)
        :param beta_end: Ending value of beta
        :param device: torch device
        """
        super().__init__()
        self.timesteps = timesteps
        self.betas = self.linear_beta_schedule(timesteps=timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.noise_pred_scalings = (1. - self.alphas) / self.sqrt_one_minus_alphas_cumprod

    def q_sample(self, x_start, t, noise=None):
        """
        扩散过程
        :param x_start: 初始图像
        :param t: 时间步
        :param noise: 噪声
        :return: 加噪后的x
        """
        noise = noise if noise is not None else torch.randn_like(x_start)

        return (self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise).to(x_start.dtype)

    def predict_start_from_noise(self, x_t, t, noise):
        return (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self.extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise).to(x_t.dtype)

    def p_sample(self, t, z_t, model, cond_embed):
        t_torch = torch.full((z_t.shape[0],), t, dtype=torch.long, device=z_t.device)
        noise_t = model(z_t, t_torch, cond_embed)
        z_t_1 = (z_t - ((1 - self.alphas[t]) / torch.sqrt(1.0 - self.alphas_cumprod[t]) * noise_t))
        if t > 1:
            sigma_t = torch.sqrt(self.betas[t])
            noise = torch.randn_like(z_t)
            z_t_1 = z_t_1 + sigma_t * noise
        return z_t_1

    @staticmethod
    def sigmoid_beta_schedule(timesteps=1000, start=-3, end=3, tau=1, clamp_min=1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        :param timesteps: 总时间步
        :param start: sigmoid 函数起始值
        :param end:
        :param tau: Sigmoid 函数的缩放因子
        :param clamp_min: betas 值进行裁剪的最小值 没用到？？？
        :return: betas
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    @staticmethod
    def linear_beta_schedule(timesteps=1000):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    @staticmethod
    def extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.to(t.device).gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
