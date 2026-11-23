"""Artery Detection & Segmentation Inference Module.

Owner: Srujan (Module 1)
Exposes the public function `detect(frame) -> mask`.

Post-Processing:
  1. Threshold sigmoid probability map (p >= 0.5)
  2. Select Largest Connected Component (LCC) to discard isolated acoustic speckles
  3. Fill internal holes using binary morphological filling
  4. Smooth lumen boundary using elliptical morphological closing/opening
  5. Returns strictly 256x256 uint8 array {0, 255} (or all zeros if no artery detected)
"""

import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch

# Setup module paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import UNet

try:
    from src.preprocess import preprocess as raw_preprocess
except ImportError:
    raw_preprocess = None


_LOADED_MODEL: Optional[torch.nn.Module] = None
_LOADED_DEVICE: Optional[torch.device] = None


def get_default_weights_path() -> str:
    """Find the default unet_best.pt checkpoint location."""
    paths = [
        os.path.join(current_dir, "models", "unet_best.pt"),
        os.path.join(current_dir, "best_unet.pth"),
        os.path.join(parent_dir, "detection", "models", "unet_best.pt"),
        os.path.join(parent_dir, "detection", "best_unet.pth"),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def load_detection_model(
    weights_path: Optional[str] = None,
    device_str: str = "auto",
) -> Tuple[torch.nn.Module, torch.device]:
    """Load or return cached U-Net model."""
    global _LOADED_MODEL, _LOADED_DEVICE

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if _LOADED_MODEL is not None and _LOADED_DEVICE == device and weights_path is None:
        return _LOADED_MODEL, _LOADED_DEVICE

    if weights_path is None:
        weights_path = get_default_weights_path()

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}. Train the model first.")

    model = UNet(in_channels=1, out_channels=1, base_channels=16).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    _LOADED_MODEL = model
    _LOADED_DEVICE = device
    return model, device


def post_process_mask(binary_mask: np.ndarray, min_area_px: int = 40) -> np.ndarray:
    """Post-processing: Largest Connected Component (LCC) + Hole Filling + Boundary Smoothing.

    Args:
        binary_mask: 2D uint8 array {0, 255}.
        min_area_px: Minimum lumen pixel area required to consider an artery detected.

    Returns:
        Clean binary mask strictly {0, 255}. Returns all zeros if no valid artery found.
    """
    if np.count_nonzero(binary_mask == 255) < min_area_px:
        return np.zeros_like(binary_mask)

    # 1. Largest Connected Component (LCC)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask)

    max_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for lbl in range(2, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = lbl

    if max_area < min_area_px:
        return np.zeros_like(binary_mask)

    lcc_mask = np.zeros_like(binary_mask, dtype=bool)
    lcc_mask[labels == max_label] = True

    # 2. Fill internal holes (acoustic dropouts inside the blood-filled lumen)
    filled = ndi.binary_fill_holes(lcc_mask)

    # 3. Morphological smoothing (elliptical structuring element)
    filled_uint8 = filled.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(filled_uint8, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return smoothed


def detect(
    frame: np.ndarray,
    weights_path: Optional[str] = None,
    threshold: float = 0.5,
    post_process: bool = True,
) -> np.ndarray:
    """Predict carotid artery lumen binary mask from an ultrasound frame.

    Args:
        frame: 2D or 3D numpy array. Accepts:
               - 256x256 float32/float64 with values in [0.0, 1.0]
               - 256x256 uint8 with values in [0, 255]
               - Raw ultrasound frame of any resolution (will be preprocessed)
        weights_path: Optional custom path to model weights.
        threshold: Sigmoid probability threshold (default: 0.5).
        post_process: Whether to apply LCC selection, hole filling, and smoothing.

    Returns:
        mask: 256x256 uint8 numpy array with values strictly {0, 255}.
              0 = background, 255 = artery lumen.
    """
    if frame is None:
        raise ValueError("Input frame is None!")

    # Standardize input to 256x256 float32 in [0.0, 1.0]
    if frame.shape != (256, 256):
        if raw_preprocess is not None:
            proc_frame = raw_preprocess(frame, size=(256, 256))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            proc_frame = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    else:
        if frame.dtype == np.uint8:
            proc_frame = frame.astype(np.float32) / 255.0
        else:
            proc_frame = frame.astype(np.float32)

    model, device = load_detection_model(weights_path=weights_path)

    # Tensor conversion (1, 1, 256, 256)
    tensor_in = torch.from_numpy(proc_frame).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.sigmoid(logits)

    prob_map = probs.squeeze().cpu().numpy()
    raw_binary = (prob_map >= threshold).astype(np.uint8) * 255

    if post_process:
        final_mask = post_process_mask(raw_binary)
    else:
        final_mask = raw_binary

    assert final_mask.shape == (256, 256), f"Output shape {final_mask.shape} != (256, 256)"
    assert final_mask.dtype == np.uint8, f"Output dtype {final_mask.dtype} != uint8"

    return final_mask

