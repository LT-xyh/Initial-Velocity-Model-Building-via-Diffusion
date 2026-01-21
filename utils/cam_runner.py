import torch
import torch.nn as nn


class SingleStepDiffusionCAMRunner(nn.Module):
    """
    Wraps conditional diffusion parts to produce a scalar for Grad-CAM.
    Forward returns per-sample scalar s = -MAE(v_hat, v_gt).
    """

    def __init__(self, cond_encoder, noise_model, scheduler, vae, timestep: int, seed: int,
                 use_posterior_mean: bool = False):
        super().__init__()
        self.cond_encoder = cond_encoder
        self.noise_model = noise_model
        self.scheduler = scheduler
        self.vae = vae
        self.timestep = int(timestep)
        self.seed = int(seed)
        self.use_posterior_mean = bool(use_posterior_mean)
        self._fixed_eps = None

    def _get_fixed_eps(self, shape, device, dtype):
        if self._fixed_eps is not None:
            if self._fixed_eps.shape == shape and self._fixed_eps.device == device and self._fixed_eps.dtype == dtype:
                return self._fixed_eps
        # Make a deterministic eps on the target device/dtype.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(self.seed)
            eps = torch.randn(shape, device=device, dtype=dtype)
        self._fixed_eps = eps
        return eps

    def _extract_alpha_bar(self, timesteps, x_shape, device, dtype):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        alpha_bar = alphas_cumprod[timesteps]
        return alpha_bar.view(-1, *([1] * (len(x_shape) - 1)))

    def _forward_impl(self, v_gt, migrated_image, rms_vel, horizon, well_log, return_vhat: bool):
        cond_dict = {"migrated_image": migrated_image, "rms_vel": rms_vel, "horizon": horizon, "well_log": well_log, }
        cond = self.cond_encoder(cond_dict)["s16"]

        posterior = self.vae.encode(v_gt)
        if self.use_posterior_mean:
            if hasattr(posterior, "mode"):
                z0 = posterior.mode()
            else:
                z0 = posterior.mean
        else:
            with torch.random.fork_rng(devices=[v_gt.device] if v_gt.device.type == "cuda" else []):
                torch.manual_seed(self.seed + 1337)
                z0 = posterior.sample()

        device = z0.device
        dtype = z0.dtype
        B = z0.shape[0]
        t_batch = torch.full((B,), self.timestep, device=device, dtype=torch.long)
        eps = self._get_fixed_eps(z0.shape, device, dtype)

        x_t = self.scheduler.add_noise(z0, eps, t_batch)
        eps_pred = self.noise_model(x_t, t_batch, cond)

        alpha_bar_t = self._extract_alpha_bar(t_batch, z0.shape, device, dtype)
        z0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t + 1e-8)
        v_hat = self.vae.decode(z0_hat)

        mae = (v_hat - v_gt).abs().mean(dim=(1, 2, 3))
        scalar = -mae
        if return_vhat:
            return v_hat, scalar
        return scalar

    def forward(self, v_gt, migrated_image, rms_vel, horizon, well_log):
        return self._forward_impl(v_gt, migrated_image, rms_vel, horizon, well_log, return_vhat=False)

    def forward_with_outputs(self, v_gt, migrated_image, rms_vel, horizon, well_log):
        v_hat, scalar = self._forward_impl(v_gt, migrated_image, rms_vel, horizon, well_log, return_vhat=True)
        return v_hat, scalar
