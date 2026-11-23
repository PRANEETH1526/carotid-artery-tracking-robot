"""PyTorch Dataset Loader for Carotid Artery Ultrasound Segmentation.

Owner: Srujan (Module 1)

DATASET DESIGN DECISION:
The images in `dataset/processed/{train,val,test}/images/` are ALREADY preprocessed
by `src/preprocess.py` (256x256 single-channel grayscale, median-denoised, and
CLAHE contrast enhanced). Therefore, we DO NOT re-run `src.preprocess.preprocess()`
during dataset loading to avoid double-processing. We simply divide the uint8 pixel
values by 255.0 to yield float32 tensors in [0.0, 1.0].
"""

import os
import glob
import random
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CarotidDataset(Dataset):
    """Loads preprocessed 256x256 ultrasound images and binary lumen masks.

    Applies synchronized data augmentations exclusively to the training split.
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        is_train: bool = False,
        augment: bool = False,
        seed: int = 42,
    ):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.is_train = is_train
        self.augment = augment and is_train
        self.rng = random.Random(seed)

        assert len(self.img_paths) == len(self.mask_paths), (
            f"Mismatch between images ({len(self.img_paths)}) and masks ({len(self.mask_paths)})"
        )
        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

    def __len__(self) -> int:
        return len(self.img_paths)

    def _apply_augmentation(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric and intensity augmentations identically to image and mask."""
        # 1. Random horizontal flip (p=0.5)
        if self.rng.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # 2. Random rotation (-10 to +10 degrees)
        if self.rng.random() > 0.5:
            angle = self.rng.uniform(-10.0, 10.0)
            M = cv2.getRotationMatrix2D((128, 128), angle, 1.0)
            img = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (256, 256), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 3. Small spatial shift (+/- 8 pixels in x and y)
        if self.rng.random() > 0.5:
            dx = self.rng.uniform(-8.0, 8.0)
            dy = self.rng.uniform(-8.0, 8.0)
            M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M_trans, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M_trans, (256, 256), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 4. Intensity/brightness jitter (+/- 10%) on image only
        if self.rng.random() > 0.5:
            factor = self.rng.uniform(0.9, 1.1)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Error reading image or mask at {img_path}")

        if self.augment:
            img, mask = self._apply_augmentation(img, mask)

        # Convert image to float32 tensor in [0.0, 1.0] (1, 256, 256)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        # Convert mask to binary float32 tensor strictly {0.0, 1.0} (1, 256, 256)
        mask_binary = (mask >= 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return img_tensor, mask_tensor
