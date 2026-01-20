import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def _load_tensor(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    raise TypeError(f"Unsupported data type at {path}: {type(obj)}")


def _select_sample(t, sample_idx):
    if t.ndim == 4:  # B, C, H, W
        t = t[sample_idx]
        if t.shape[0] == 1:
            t = t[0]
    elif t.ndim == 3:  # B, H, W
        t = t[sample_idx]
    elif t.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")
    return t.detach().cpu().float().numpy()


def _resolve_dataset_path(root, dataset):
    root = Path(root)
    if root.name == dataset:
        return root
    candidate = root / dataset
    if candidate.exists():
        return candidate
    return root


def _pearson(a, b):
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def _load_from_folder(folder, batch_idx, sample_idx):
    path = Path(folder) / f"{batch_idx}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return _select_sample(_load_tensor(str(path)), sample_idx)


def main():
    parser = argparse.ArgumentParser(description="Well match degree visualization.")
    parser.add_argument("--dataset", default="CurveVelA")
    parser.add_argument("--diffusion-root", default="logs/ddpm_diffusion/test_results/test_260114")
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--well-threshold", type=float, default=-1.0)
    parser.add_argument("--max-wells", type=int, default=4)
    parser.add_argument("--save-dir", default="images/well_match")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Baseline in name=path format. Repeat for multiple baselines.",
    )
    parser.add_argument("--gt-path", default=None, help="Optional override for ground-truth folder.")
    parser.add_argument("--diffusion-name", default="Diffusion")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Diffusion paths
    diff_root = Path(args.diffusion_root)
    diff_dataset_root = _resolve_dataset_path(diff_root, args.dataset)
    gt_root = Path(args.gt_path) if args.gt_path else diff_dataset_root / "GroundTruth"
    diff_recon_root = diff_dataset_root / "Recon"
    diff_well_root = diff_dataset_root / "well_log"

    gt = _load_from_folder(gt_root, args.batch_idx, args.sample_idx)
    diff = _load_from_folder(diff_recon_root, args.batch_idx, args.sample_idx)
    well_log = _load_from_folder(diff_well_root, args.batch_idx, args.sample_idx)

    # Baselines
    baselines = []
    for item in args.baseline:
        if "=" not in item:
            raise ValueError("Baseline must be in name=path format.")
        name, path = item.split("=", 1)
        dataset_path = _resolve_dataset_path(path, args.dataset)
        pred = _load_from_folder(dataset_path, args.batch_idx, args.sample_idx)
        baselines.append((name, pred))

    # Prepare mask and wells
    mask = well_log >= args.well_threshold
    well_cols = np.where(mask.any(axis=0))[0].tolist()
    if not well_cols:
        raise ValueError("No well positions found. Check --well-threshold or well_log.")
    if args.max_wells > 0:
        well_cols = well_cols[: args.max_wells]

    # Compute correlation over all well positions
    mask_vals = mask.astype(bool)
    gt_vals = gt[mask_vals]
    method_preds = [(args.diffusion_name, diff)] + baselines
    method_cc = {
        name: _pearson(gt_vals, pred[mask_vals]) for name, pred in method_preds
    }

    # Figure 1: well positions + curves at selected wells
    n_rows = 1 + len(well_cols)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.2 * n_rows), squeeze=False)
    ax0 = axes[0, 0]
    ax0.imshow(mask.astype(np.float32), cmap="gray", aspect="auto")
    ax0.set_title("Well positions (from Diffusion well_log)")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Depth")

    depth = np.arange(gt.shape[0])
    for i, col in enumerate(well_cols, start=1):
        ax = axes[i, 0]
        ax.plot(depth, gt[:, col], label="GroundTruth", linewidth=2)
        for name, pred in method_preds:
            label = f"{name} (cc={method_cc[name]:.3f})"
            ax.plot(depth, pred[:, col], label=label, linewidth=1.6)
        ax.set_title(f"Well column x={col}")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Velocity (normalized)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig_path = save_dir / f"well_match_curves_{args.dataset}_{args.batch_idx}_{args.sample_idx}.svg"
    fig.savefig(fig_path)

    # Figure 2: scatter plots vs GT at well positions
    n_methods = len(method_preds)
    fig2, axes2 = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), squeeze=False)
    for i, (name, pred) in enumerate(method_preds):
        ax = axes2[0, i]
        ax.scatter(gt_vals, pred[mask_vals], s=8, alpha=0.4)
        min_v = min(gt_vals.min(), pred[mask_vals].min())
        max_v = max(gt_vals.max(), pred[mask_vals].max())
        ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
        ax.set_title(f"{name} (cc={method_cc[name]:.3f})")
        ax.set_xlabel("GroundTruth")
        ax.set_ylabel("Prediction")
        ax.grid(True, alpha=0.2)

    fig2.tight_layout()
    scatter_path = save_dir / f"well_match_scatter_{args.dataset}_{args.batch_idx}_{args.sample_idx}.svg"
    fig2.savefig(scatter_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)

    print(f"Saved: {fig_path}")
    print(f"Saved: {scatter_path}")


if __name__ == "__main__":
    main()


    #conda activate seg
# python scripts/visualize_well_match.py --dataset CurveVelA --batch-idx 0 --sample-idx 0 --diffusion-root logs/ddpm_diffusion/test_results/test_260114 --baseline InversionNet=logs/baseline/inversion_net --baseline SVInvNet=logs/baseline/sv_inv_net --baseline VelocityGAN=logs/baseline/velocity_gan --baseline Dix=logs/baseline/dix/251015/images/CurveVelA --save-dir images/well_match --show
