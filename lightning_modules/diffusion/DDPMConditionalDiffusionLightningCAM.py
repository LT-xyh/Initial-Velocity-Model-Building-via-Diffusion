import os
from typing import Dict, List, Optional

import numpy as np
import torch
from contextlib import nullcontext

from lightning_modules.diffusion.DDPMConditionalDiffusionLightning import DDPMConditionalDiffusionLightning
from utils.cam_runner import SingleStepDiffusionCAMRunner
from utils.gradcam_utils import prepare_base_image, run_gradcam, save_cam_overlay, save_image


class DDPMConditionalDiffusionLightningCAM(DDPMConditionalDiffusionLightning):
    """
    Minimal-intrusion CAM wrapper: reuse existing test/validation logic, add optional CAM generation.
    """

    def __init__(self, conf, cam_cfg: Optional[Dict] = None):
        super().__init__(conf)
        self.cam_cfg = self._build_cam_cfg(cam_cfg or {})

    @staticmethod
    def _build_cam_cfg(user_cfg: Dict) -> Dict:
        defaults = {
            "enabled": False,
            "timestep": 500,
            "seed": 1234,
            "max_batches": 1,
            "sample_indices": [0],
            "out_dir": None,
            "save_inputs": True,
            "save_pred_gt": True,
            "save_npy": False,
            "use_posterior_mean": False,
            "disable_amp": True,
        }
        out = defaults
        out.update(user_cfg)
        return out

    def test_step(self, batch, batch_idx):
        # Keep the original test behavior untouched.
        batch_for_super = {k: v for k, v in batch.items()}
        out = super().test_step(batch_for_super, batch_idx)

        if not self.cam_cfg.get("enabled", False):
            return out
        if batch_idx >= int(self.cam_cfg.get("max_batches", 1)):
            return out

        self.compute_cam(batch, batch_idx)
        return out

    def _get_cam_layers(self) -> Dict[str, torch.nn.Module]:
        cond_enc = self.ldm_cond_encoder.cond_encoder
        return {
            "rms": cond_enc["rms_vel"].proj,
            "migrated": cond_enc["migrated_image"].out,
            "horizon": cond_enc["horizon"].head,
            "well": cond_enc["well_log"].out,
        }

    def compute_cam(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self.eval()
        device = self.device

        depth_vel = batch.get("depth_vel")
        migrated_image = batch.get("migrated_image")
        rms_vel = batch.get("rms_vel")
        horizon = batch.get("horizon")
        well_log = batch.get("well_log")

        if any(x is None for x in [depth_vel, migrated_image, rms_vel, horizon, well_log]):
            return

        t = int(self.cam_cfg["timestep"])
        seed = int(self.cam_cfg["seed"])
        out_dir = self.cam_cfg["out_dir"] or os.path.join(self.conf.testing.test_save_dir, "cam")
        os.makedirs(out_dir, exist_ok=True)
        batch_dir = os.path.join(out_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)

        sample_indices: List[int] = list(self.cam_cfg.get("sample_indices", [0]))
        sample_indices = [i for i in sample_indices if 0 <= i < depth_vel.shape[0]]
        if len(sample_indices) == 0:
            sample_indices = [0]

        runner = SingleStepDiffusionCAMRunner(
            cond_encoder=self.ldm_cond_encoder,
            noise_model=self.ldm.noise_model,
            scheduler=self.ldm.scheduler,
            vae=self.vae,
            timestep=t,
            seed=seed,
            use_posterior_mean=self.cam_cfg.get("use_posterior_mean", False),
        ).to(device)
        runner.eval()

        use_cuda = device.type == "cuda"
        disable_amp = bool(self.cam_cfg.get("disable_amp", True))
        layers = self._get_cam_layers()

        for sidx in sample_indices:
            v_gt = depth_vel[sidx:sidx + 1]
            mig = migrated_image[sidx:sidx + 1]
            rms = rms_vel[sidx:sidx + 1]
            hor = horizon[sidx:sidx + 1]
            well = well_log[sidx:sidx + 1]

            # Save inputs and prediction if requested.
            if self.cam_cfg.get("save_inputs", True):
                save_image(prepare_base_image(mig), os.path.join(
                    batch_dir, f"t{t}_seed{seed}_input_migrated_sample{sidx}.png"))
                save_image(prepare_base_image(rms), os.path.join(
                    batch_dir, f"t{t}_seed{seed}_input_rms_sample{sidx}.png"))
                save_image(prepare_base_image(hor), os.path.join(
                    batch_dir, f"t{t}_seed{seed}_input_horizon_sample{sidx}.png"))
                save_image(prepare_base_image(well), os.path.join(
                    batch_dir, f"t{t}_seed{seed}_input_well_sample{sidx}.png"))

            with torch.set_grad_enabled(True):
                autocast_ctx = torch.cuda.amp.autocast(enabled=not disable_amp) if use_cuda else nullcontext()
                with autocast_ctx:
                    for name, layer in layers.items():
                        runner.zero_grad(set_to_none=True)
                        inputs = (v_gt, mig, rms, hor, well)
                        cam_map = run_gradcam(runner, layer, inputs, use_cuda=use_cuda)[0]

                        base = {
                            "rms": rms,
                            "migrated": mig,
                            "horizon": hor,
                            "well": well,
                        }[name]
                        base_img = prepare_base_image(base)
                        out_path = os.path.join(
                            batch_dir,
                            f"t{t}_seed{seed}_targetGlobalMAE_branch{name}_sample{sidx}.png",
                        )
                        save_cam_overlay(base_img, cam_map, out_path)

                        if self.cam_cfg.get("save_npy", False):
                            npy_path = out_path.replace(".png", ".npy")
                            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                            np.save(npy_path, cam_map)

            if self.cam_cfg.get("save_pred_gt", True):
                with torch.set_grad_enabled(True):
                    v_hat, _ = runner.forward_with_outputs(v_gt, mig, rms, hor, well)
                save_image(prepare_base_image(v_hat), os.path.join(
                    batch_dir, f"t{t}_seed{seed}_pred_sample{sidx}.png"))
                save_image(prepare_base_image(v_gt), os.path.join(
                    batch_dir, f"t{t}_seed{seed}_gt_sample{sidx}.png"))
