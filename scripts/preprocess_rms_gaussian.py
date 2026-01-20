import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("kernel_size must be an odd integer greater than 1.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def smooth_array(arr: np.ndarray, kernel: torch.Tensor, pad: int, noise_std: float) -> np.ndarray:
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0.")
    if arr.ndim == 2:
        x = torch.from_numpy(arr).to(dtype=torch.float32, device=kernel.device)
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        x = x.unsqueeze(0).unsqueeze(0)
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y = F.conv2d(x, kernel.view(1, 1, *kernel.shape))
        return y.squeeze(0).squeeze(0).cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] < 1:
            raise ValueError(f"Expected non-empty first dim, got shape {arr.shape}")
        out = np.empty_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i] = smooth_array(arr[i], kernel, pad, noise_std)
        return out
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def list_datasets(root_dir: str, datasets: list[str] | None) -> list[str]:
    if datasets:
        return datasets
    candidates = []
    for name in sorted(os.listdir(root_dir)):
        dataset_dir = os.path.join(root_dir, name)
        if not os.path.isdir(dataset_dir):
            continue
        if os.path.isdir(os.path.join(dataset_dir, "rms_vel")) or os.path.isdir(
            os.path.join(dataset_dir, "rms_vel_raw")
        ):
            candidates.append(name)
    return candidates


def archive_rms_dir(dataset_dir: str, rms_dirname: str, rms_raw_dirname: str) -> tuple[str, str]:
    rms_dir = os.path.join(dataset_dir, rms_dirname)
    rms_raw_dir = os.path.join(dataset_dir, rms_raw_dirname)
    if os.path.isdir(rms_raw_dir):
        return rms_dir, rms_raw_dir
    if not os.path.isdir(rms_dir):
        raise FileNotFoundError(f"Missing rms directory: {rms_dir}")
    shutil.move(rms_dir, rms_raw_dir)
    return rms_dir, rms_raw_dir


def process_dataset(
    dataset_dir: str,
    kernel_size: int,
    sigma: float,
    overwrite: bool,
    noise_std: float,
    rms_dirname: str,
    rms_raw_dirname: str,
) -> None:
    rms_dir, rms_raw_dir = archive_rms_dir(dataset_dir, rms_dirname, rms_raw_dirname)
    os.makedirs(rms_dir, exist_ok=True)
    device = torch.device("cpu")
    kernel = gaussian_kernel2d(kernel_size, sigma, device)
    pad = kernel_size // 2
    for filename in sorted(os.listdir(rms_raw_dir)):
        if not filename.endswith(".npy"):
            continue
        src_path = os.path.join(rms_raw_dir, filename)
        dst_path = os.path.join(rms_dir, filename)
        if os.path.exists(dst_path) and not overwrite:
            continue
        arr = np.load(src_path)
        smoothed = smooth_array(arr, kernel, pad, noise_std)
        np.save(dst_path, smoothed.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaussian smooth OpenFWI rms_vel and archive originals.")
    parser.add_argument("--root-dir", default="data/openfwi", help="Root directory of OpenFWI datasets.")
    parser.add_argument("--datasets", nargs="*", help="Dataset names to process. Default: auto-detect.")
    parser.add_argument("--kernel-size", type=int, default=5, help="Odd kernel size.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Additive Gaussian noise std before smoothing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing smoothed files.")
    parser.add_argument("--rms-dirname", default="rms_vel", help="Smoothed rms directory name.")
    parser.add_argument("--rms-raw-dirname", default="rms_vel_raw", help="Archived rms directory name.")
    args = parser.parse_args()

    datasets = list_datasets(args.root_dir, args.datasets)
    if not datasets:
        raise RuntimeError(f"No datasets found under {args.root_dir}")
    for dataset in datasets:
        dataset_dir = os.path.join(args.root_dir, dataset)
        process_dataset(
            dataset_dir,
            args.kernel_size,
            args.sigma,
            args.overwrite,
            args.noise_std,
            args.rms_dirname,
            args.rms_raw_dirname,
        )


if __name__ == "__main__":
    main()
    # python scripts/preprocess_rms_gaussian.py --root-dir data/openfwi --kernel-size 5 --sigma 2.5 --noise-std 20 --datasets CurveVelA --overwrite
    # python scripts/preprocess_rms_gaussian.py --root-dir data/openfwi --kernel-size 5 --sigma 2.5 --noise-std 20 --datasets CurveFaultA

