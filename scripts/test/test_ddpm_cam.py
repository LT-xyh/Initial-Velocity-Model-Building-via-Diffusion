import os
from datetime import datetime

import torch
from omegaconf import OmegaConf

from lightning_modules.diffusion.DDPMConditionalDiffusionLightningCAM import DDPMConditionalDiffusionLightningCAM
from scripts.test.basetest import base_test


def test_ddpm_cam(dataset_name):
    current_date = datetime.now()
    date_str = current_date.strftime("%y%m%d")
    torch.set_float32_matmul_precision('medium')

    conf = OmegaConf.load('configs/ddpm_cond_diffusion.yaml')
    conf.datasets.dataset_name = [dataset_name, ]

    conf.testing.test_save_dir = f'{conf.testing.test_save_dir}/cam_test_{date_str}/{dataset_name}'
    conf.testing.ckpt_path = 'logs/ddpm_diffusion/tensorboard/260120-16base_cond-CA_rms-smooth/checkpoints/epoch_28-loss0.880.ckpt'
    conf.training.logging.log_version = f"cam_test/{date_str}_{dataset_name}"

    cam_cfg = {"enabled": True, "timestep": 500, "seed": 1234, "max_batches": 1, "sample_indices": [0],
               "out_dir": os.path.join(conf.testing.test_save_dir, "cam"), "save_inputs": True, "save_pred_gt": True,
               "save_npy": True, "use_posterior_mean": False, "disable_amp": True, }

    model = DDPMConditionalDiffusionLightningCAM.load_from_checkpoint(conf.testing.ckpt_path, conf=conf,
                                                                      cam_cfg=cam_cfg)
    base_test(model, conf, fast_run=False)


"""
How to run CAM:
    Update the checkpoint path inside test_ddpm_cam.py if needed.

    Run: python scripts/test/test_ddpm_cam.py

Outputs:
    Saved under conf.testing.test_save_dir/cam/batch_{idx}/.
    CAM overlays named like t{t}_seed{seed}_targetGlobalMAE_branch{rms|migrated|horizon|well}_sample{idx}.png.
    Inputs and pred/gt are saved alongside, and raw CAM arrays are saved as .npy when enabled.
    Notes:

CAM uses a fixed timestep and fixed seed, and runs the single‑step UNet forward with gradients enabled (no torch.no_grad() around VAE decode in the CAM path).
The runner uses deterministic VAE sampling (seeded) if use_posterior_mean=False; you can set use_posterior_mean=True in cam_cfg for fully deterministic z0.
"""

if __name__ == '__main__':
    for dataset_name in ['FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB']:
        print(f'\n\n{dataset_name}\n')
        test_ddpm_cam(dataset_name)
