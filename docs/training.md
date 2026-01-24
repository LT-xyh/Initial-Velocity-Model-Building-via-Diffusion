# Training and evaluation

This repo uses PyTorch Lightning. Training and testing scripts live in `scripts/trains/` and `scripts/test/` and load YAML configs from `configs/`.

## Configuration structure
Most configs share a common layout:
- `datasets`: dataset names, modalities (`use_data`), normalization, and target shapes
- `autoencoder_conf`: autoencoder architecture or checkpoint path (diffusion models)
- `latent_diffusion`: diffusion scheduler settings (DDPM or ODE)
- `training`: optimizer, precision, dataloader settings, logging, and callbacks
- `testing`: checkpoint path and output directory for test results

Example (from `configs/ddpm_cond_diffusion.yaml`):
```
datasets:
  use_data: ['depth_vel', 'migrated_image', 'well_log', 'horizon', 'rms_vel']
  dataset_name: ['FlatVelA', 'FlatVelB', 'CurveVelA', 'CurveVelB']
  use_normalize: '-1_1'
  depth_velocity:
    shape: [1, 70, 70]

autoencoder_conf:
  autoencoder_checkpoint_path: 'path/to/autoencoder.ckpt'

latent_diffusion:
  scheduler:
    scheduler_type: 'ddpm'
    num_train_timesteps: 1000

training:
  lr: 1e-4
  use_ema: true
  max_epochs: 150
  device: [0]
  precision: 'bf16-mixed'
  dataloader:
    batch_size: 50
    num_workers: 2
```

## Training flow
### 1) Train an autoencoder
Train a KL autoencoder on `depth_vel`:
```
python scripts/trains/train_autoencoder_kl.py
```
Config: `configs/autoencoder_kl.yaml`.

If you use the non-KL autoencoder, see `scripts/trains/train_autoencoder.py` and align `datasets.use_data` with the dataset keys it expects.

### 2) Train conditional diffusion
Update `autoencoder_conf.autoencoder_checkpoint_path` in `configs/ddpm_cond_diffusion.yaml`, then run:
```
python scripts/trains/train_ddpm_cond_diffusion.py
```
This uses:
- `lightning_modules/diffusion/DDPMConditionalDiffusionLightning.py`
- `models/conditional_encoder/CondFusionPyramid70.py` for conditioning
- `models/diffusion/DiffusionConditionedUNet.py` for the diffusion core

### 3) Train baseline models
Baseline trainers include:
```
python scripts/trains/train_inversion_net.py
python scripts/trains/train_sv_inv_net.py
python scripts/trains/train_velocity_gan.py
```
Each uses its matching config in `configs/`.

### 4) Fine-tune on field cuts (optional)
`base_train_field_cut` in `scripts/trains/basetrain.py` trains on `data/data1_cut`. The fine-tuning function in `scripts/trains/train_ddpm_cond_diffusion.py` shows an example setup.

## Evaluation and testing
Test scripts live under `scripts/test/`. Most require you to set a checkpoint path in the config.
Example:
```
python scripts/test/test_ddpm_cond_diffusion.py
```
The test scripts use `conf.testing.ckpt_path` and write `.pt` outputs into `conf.testing.test_save_dir`.

## Outputs
- Logs: `logs/<model>/tensorboard/<version>/` and `logs/<model>/csv/<version>/`.
- Checkpoints: saved under the TensorBoard log directory by Lightning.
- Test outputs: `.pt` files written by `BaseLightningModule.save_batch_torch`.

## Tips
- Run scripts from the repo root to keep relative paths consistent.
- `base_train` holds out the last 10 percent of each dataset for testing, then splits the rest into train and validation.
- If you want a quick sanity check, pass `fast_run=True` to `base_train` in your training script (Lightning `fast_dev_run`).
