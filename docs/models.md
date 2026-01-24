# Models and components

This repo is organized around three main model families: autoencoders, conditional diffusion models, and baseline inversion networks.

## Autoencoders
Purpose: compress 70x70 velocity models into a latent space (typically 16x16x16) for diffusion training.

Key files:
- `models/Autoencoder/AutoencoderKLInterpolation.py`: wraps diffusers `AutoencoderKL` with interpolation layers to map 70x70 <-> 64x64.
- `models/Autoencoder/Autoencoder.py` and `models/Autoencoder/MyAE.py`: lighter-weight AE variants.
- Lightning wrappers:
  - `lightning_modules/Autoencoder/autoencoder_kl_lightning.py`: KL autoencoder with annealed KL weight and EMA.
  - `lightning_modules/Autoencoder/AutoencoderLightning.py`: AE variant that expects `batch['model']`.
  - `lightning_modules/Autoencoder/AutoencoderLightning.py` and `lightning_modules/Autoencoder/autoencoder_kl_lightning.py` handle training/validation metrics and logging.

## Conditional encoders
Purpose: turn multiple geophysical constraints into a fused multi-scale conditioning pyramid.

Key files:
- `models/conditional_encoder/CondFusionPyramid70.py`: fuses modality features and outputs multi-scale maps: `s70`, `s64`, `s32`, `s16`, plus modality weights.
- `models/conditional_encoder/*Encoder_70x70.py`: per-modality encoders (migrated image, horizon, well log, RMS velocity).

The conditioning pyramid is consumed by diffusion models through cross-attention tokens and optional adapter residuals.

## Diffusion models
Purpose: generate latent velocity models conditioned on the fused constraints.

Key files:
- `models/diffusion/DiffusionConditionedUNet.py`: conditional UNet (cross-attn, concat, or adapter modes) plus `LatentConditionalDiffusion` training/sampling wrapper.
- `models/diffusion/latent_cond_diffusion.py`: an alternative conditioned UNet wrapper that supports pyramid adapters (`s16/s32/s64/s70`).
- `models/schedule/ProbabilityFlowODEScheduler.py`: custom scheduler for DDIM / probability-flow ODE sampling.

Lightning wrappers:
- `lightning_modules/diffusion/DDPMConditionalDiffusionLightning.py`
- `lightning_modules/diffusion/LatentConditionalDiffusionLightning.py`
- `lightning_modules/diffusion/CondLatentDiffusionLightning.py`

## Baseline inversion models
Purpose: provide non-diffusion baselines for comparison.

Key files:
- `models/baselines/InversionNet.py`
- `models/baselines/SVInvNet.py`
- `models/baselines/VelocityGAN.py`
- `models/baselines/dix.py`

Lightning wrappers live in `lightning_modules/baselines_lightning/`.

## Metrics and visualization
- `utils/metrics.py`: PSNR and SSIM tracking (used by `BaseLightningModule`).
- `utils/visualize.py`: helper plotting utilities used during testing and logging.
