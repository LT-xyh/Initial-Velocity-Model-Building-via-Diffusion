# Data and datasets

This codebase expects datasets stored as per-sample `.npy` files in modality-specific folders. The dataset classes live in `data/`:
- `data/dataset_openfwi.py`: OpenFWI-style synthetic datasets
- `data/dataset_field.py`: field data (full sections)
- `data/dataset_field_cut.py`: field data (cut/patch subsets)

## OpenFWI directory layout
The OpenFWI loader expects:
```
data/openfwi/
  FlatVelA/
    depth_vel/
      0.npy
      1.npy
      ...
    migrated_image/
    well_log/
    horizon/
    rms_vel/
    time_vel/           # optional, only if you include it in use_data
  FlatVelB/
  CurveVelA/
  CurveVelB/
  CurveFaultA/          # used by fault tests
```

Dataset names are configured in `configs/*.yaml` under `datasets.dataset_name` (for example `FlatVelA`, `CurveVelB`).

## Field data layouts
Two loaders are included for field data:
```
data/Field_data_70x70/
  depth_vel/
  migrated_image/
  well_log/
  horizon/
  rms_vel/

# for fine-tuning / cut datasets

data/data1_cut/
  depth_vel/
  migrated_image/
  well_log/
  horizon/
  rms_vel/
```

## Modality keys
The dataset returns a dictionary with the following keys (depending on `datasets.use_data`):

| Key             | Meaning                          | Typical shape | Notes |
| ---             | ---                              | ---           | ---   |
| `depth_vel`     | depth-domain velocity model      | 1x70x70       | main target for reconstruction |
| `time_vel`      | time-domain velocity             | 1x70x70       | optional |
| `migrated_image`| migrated seismic image           | 1x70x70       | conditioning input |
| `well_log`      | well-log mask or curve           | 1x70x70       | conditioning input |
| `horizon`       | horizon picks                    | 1x70x70       | conditioning input |
| `rms_vel`       | RMS velocity                     | 1x70x70       | conditioning input |

The encoders in `models/conditional_encoder/` assume single-channel inputs with spatial size 70x70.

## Normalization
Normalization is applied inside the dataset classes via `datasets.use_normalize`:
- `'01'`: normalize to [0, 1]
- `'-1_1'`: normalize to [-1, 1]
- `None` (OpenFWI only): no normalization

Each dataset class has hard-coded min/max ranges per modality:
- `data/dataset_openfwi.py`
- `data/dataset_field.py`
- `data/dataset_field_cut.py`

If your data ranges differ, update `normalize_max_min` in the appropriate dataset class or disable normalization.

## RMS preprocessing
There is a helper to smooth RMS velocity files and archive the originals:
```
python scripts/preprocess_rms_gaussian.py --root-dir data/openfwi --kernel-size 5 --sigma 2.5 --datasets CurveVelA --overwrite
```
This moves `rms_vel` to `rms_vel_raw` and writes smoothed files to a new `rms_vel` folder.
