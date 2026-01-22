import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
import inspect
from torch import nn


class ScalarTarget:
    """Selects a single scalar from model output for Grad-CAM."""

    def __init__(self, index: int = 0):
        self.index = int(index)

    def __call__(self, model_output):
        if model_output.ndim == 0:
            return model_output
        return model_output[self.index]


def _to_numpy_2d(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.mean(axis=0)
    return arr


def _normalize_minmax(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < eps:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn + eps)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image(arr: np.ndarray, out_path: str, cmap: str = "jet", title: Optional[str] = None):
    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(5, 5))
    plt.imshow(arr, cmap=cmap, aspect="auto")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_cam_overlay(base_img: np.ndarray, cam: np.ndarray, out_path: str, alpha: float = 0.5):
    _ensure_dir(os.path.dirname(out_path))
    base = _normalize_minmax(base_img)
    cam_n = _normalize_minmax(cam)
    plt.figure(figsize=(5, 5))
    plt.imshow(base, cmap="gray", aspect="auto")
    plt.imshow(cam_n, cmap="jet", alpha=alpha, aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


class _MultiInputWrapper(nn.Module):
    def __init__(self, model: nn.Module, fixed_inputs, primary_index: int):
        super().__init__()
        self.model = model
        self.fixed_inputs = tuple(fixed_inputs)
        self.primary_index = int(primary_index)

    def forward(self, primary_tensor: torch.Tensor):
        if getattr(primary_tensor, "is_inference", None) and primary_tensor.is_inference():
            primary_tensor = primary_tensor.clone()
        inputs = list(self.fixed_inputs)
        device = primary_tensor.device
        for i, value in enumerate(inputs):
            if i == self.primary_index:
                continue
            if isinstance(value, torch.Tensor) and value.device != device:
                inputs[i] = value.to(device)
            if isinstance(value, torch.Tensor) and getattr(value, "is_inference", None) and value.is_inference():
                inputs[i] = value.clone()
        inputs[self.primary_index] = primary_tensor
        return self.model(*inputs)


def _build_gradcam(model, target_layer, use_cuda: bool):
    # Handle API differences across pytorch-grad-cam versions without triggering __del__ errors.
    kwargs = {"model": model, "target_layers": [target_layer]}
    try:
        sig = inspect.signature(GradCAM.__init__)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if "device" in params or has_varkw:
            kwargs["device"] = torch.device("cuda" if use_cuda else "cpu")
        elif "use_cuda" in params:
            kwargs["use_cuda"] = use_cuda

    return GradCAM(**kwargs)


def _prepare_cam_inputs(model, input_tensors, primary_input, primary_index: int):
    if isinstance(input_tensors, (tuple, list)):
        if primary_input is None:
            primary_input = input_tensors[primary_index]
        model = _MultiInputWrapper(model, input_tensors, primary_index)
    return model, primary_input if primary_input is not None else input_tensors


def run_gradcam(model, target_layer, input_tensors, use_cuda: bool,
                primary_input: Optional[torch.Tensor] = None, primary_index: int = 0):
    model, cam_input = _prepare_cam_inputs(model, input_tensors, primary_input, primary_index)
    cam = _build_gradcam(model, target_layer, use_cuda)
    targets = [ScalarTarget(0)]
    grayscale_cam = cam(input_tensor=cam_input, targets=targets)
    if isinstance(grayscale_cam, np.ndarray):
        return grayscale_cam
    return np.array(grayscale_cam)


def prepare_base_image(t: torch.Tensor, target_hw=(70, 70)) -> np.ndarray:
    if t.ndim == 4 and t.shape[-2:] != target_hw:
        t = F.adaptive_avg_pool2d(t, target_hw)
    return _to_numpy_2d(t)
