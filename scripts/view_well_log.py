import argparse
import os

import numpy as np
import torch

from data.dataset_openfwi import OpenFWI
from utils.visualize import save_multiple_curves, save_visualize_image


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def load_sample(args):
    use_data = ('depth_vel', 'migrated_image', 'well_log', 'horizon', 'rms_vel')
    if args.dataset == "openfwi":
        dataset = OpenFWI(
            root_dir=args.root_dir,
            use_data=use_data,
            datasets=(args.dataset_name,),
            use_normalize=args.normalize,
        )
    return dataset[args.index]


def main():
    parser = argparse.ArgumentParser(description="Inspect well_log samples (1x70x70).")
    parser.add_argument("--dataset", choices=["openfwi", "field_cut"], default="openfwi")
    parser.add_argument("--root-dir", default="data/openfwi")
    parser.add_argument("--dataset-name", default="CurveVelA")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--normalize", choices=["-1_1", "01"], default="-1_1")
    parser.add_argument("--well-threshold", type=float, default=-1.0)
    parser.add_argument("--save-dir", default="images/well_log_debug")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    sample = load_sample(args)
    os.makedirs(args.save_dir, exist_ok=True)

    depth_vel = _to_numpy(sample["depth_vel"]).squeeze()
    well_log = _to_numpy(sample["well_log"]).squeeze()

    print(f"depth_vel shape: {depth_vel.shape}")
    print(f"well_log shape: {well_log.shape}")

    save_visualize_image(
        depth_vel,
        filename=os.path.join(args.save_dir, f"depth_vel_{args.index}.svg"),
        title="Depth Velocity",
        show=args.show,
        save=True,
        cmap="jet",
        extent=None,
        figsize=(5, 5),
        use_colorbar=False,
        x_label="Length (m)",
        y_label="Depth (m)",
    )

    save_visualize_image(
        well_log,
        filename=os.path.join(args.save_dir, f"well_log_{args.index}.svg"),
        title="Well Log (values along depth)",
        show=args.show,
        save=True,
        cmap="jet",
        extent=None,
        figsize=(5, 5),
        use_colorbar=False,
        x_label="Length (m)",
        y_label="Depth (m)",
    )

    well_mask = well_log >= args.well_threshold
    save_visualize_image(
        well_mask.astype(np.float32),
        filename=os.path.join(args.save_dir, f"well_mask_{args.index}.svg"),
        title="Well Mask",
        show=args.show,
        save=True,
        cmap="gray",
        extent=None,
        figsize=(5, 5),
        use_colorbar=False,
        x_label="Length (m)",
        y_label="Depth (m)",
    )

    well_cols = np.where(well_mask.any(axis=0))[0].tolist()
    print(f"well columns: {well_cols}")
    if well_cols:
        curves = [well_log[:, col] for col in well_cols]
        labels = [f"x={col}" for col in well_cols]
        save_multiple_curves(
            curves,
            labels=labels,
            filename=os.path.join(args.save_dir, f"well_log_curves_{args.index}.svg"),
            title="Well log curves",
            x_label="Depth (m)",
            y_label="Velocity (normalized)",
            show=args.show,
            save=True,
            figsize=(6, 6),
            colors=None,
            linestyles=None,
        )


if __name__ == "__main__":
    main()
