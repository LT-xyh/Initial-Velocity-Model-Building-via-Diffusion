# ldm-velocity-model

Research code for seismic velocity model reconstruction using conditional latent diffusion and baseline inversion models. The core workflow is:
1) train an autoencoder on depth velocity, 2) encode into a latent space, 3) train a conditional diffusion model guided by multi-constraint inputs (migrated image, well log, horizon, RMS velocity). The repo also includes baseline models such as InversionNet, SVInvNet, and VelocityGAN.

## Documentation map
- data.md: dataset layout, modality keys, and normalization details
- models.md: model components and how they connect
- training.md: training, fine-tuning, and evaluation workflows

## Quickstart
1) Create a Python environment and install dependencies. At minimum the training scripts use:
   - torch, torchvision
   - lightning
   - omegaconf
   - diffusers
   - numpy, scikit-learn
   - lpips, pytorch-msssim (for perceptual and SSIM losses)
   - ignite, torchmetrics (metrics)
   - matplotlib (visualization)

2) Prepare data under `data/` (see data.md for the expected directory structure).

3) Train an autoencoder (KL version shown here):
```
python scripts/trains/train_autoencoder_kl.py
```

4) Update `configs/ddpm_cond_diffusion.yaml` with the autoencoder checkpoint path:
- `autoencoder_conf.autoencoder_checkpoint_path`

5) Train conditional diffusion:
```
python scripts/trains/train_ddpm_cond_diffusion.py
```

## Repo layout
- `configs/`: YAML configuration files for datasets, training, and model settings
- `data/`: dataset loaders and local dataset folders
- `lightning_modules/`: Lightning training wrappers for autoencoders, diffusion, and baselines
- `models/`: model definitions (autoencoders, diffusion, baselines, conditional encoders)
- `scripts/`: training, testing, preprocessing, and visualization scripts
- `utils/`: metrics and visualization helpers
- `logs/`: training outputs (created at runtime)

## Notes
- Run scripts from the repository root so relative paths and imports resolve correctly.
- Some configs reference older dataset keys (for example `model`, `migrate`, `horizens`). Ensure `datasets.use_data` matches the keys produced by the dataset class you are using.
