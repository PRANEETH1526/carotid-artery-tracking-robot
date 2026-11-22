# Artery Detection & Segmentation Module

**Owner:** Srujan  
**Component:** Carotid Artery Lumen Segmentation (U-Net)

---

## Purpose
Train, validate, and serve a 2D deep learning segmentation model (e.g., U-Net with ResNet/EfficientNet or custom CNN backbone) to segment the common carotid artery lumen from preprocessed transverse B-mode ultrasound frames.

---

## Interface Contract

### Input
- **Single Frame:** Preprocessed 2D numpy array `frame` of shape `(256, 256)` and dtype `float32` with pixel intensities in `[0.0, 1.0]`.
- Produced directly by `src.preprocess.preprocess(raw_img)`.

### Output
- **Binary Segmentation Mask:** 2D numpy array `mask` of shape `(256, 256)` and dtype `uint8` with values strictly `0` (background) and `255` (artery lumen).

### Standard Function Signature
```python
import numpy as np

def detect(frame: np.ndarray) -> np.ndarray:
    """Predict carotid artery lumen binary mask from preprocessed ultrasound frame.

    Args:
        frame: Preprocessed ultrasound image (256x256 float32, [0.0, 1.0]).

    Returns:
        mask: Binary mask (256x256 uint8, {0, 255}).
    """
    # 1. Forward pass through trained U-Net model
    # 2. Sigmoid probability threshold (e.g. > 0.5)
    # 3. Post-processing (optional largest connected component)
    # 4. Return mask (0 or 255)
    pass
```

---

## Dataset Location
- **Train Images:** `../dataset/processed/train/images/`
- **Train Masks:** `../dataset/processed/train/masks/`
- **Validation Images:** `../dataset/processed/val/images/`
- **Validation Masks:** `../dataset/processed/val/masks/`
- **Test Images:** `../dataset/processed/test/images/`
- **Test Masks:** `../dataset/processed/test/masks/`

> **Note:** Model weights (`*.pt`, `*.pth`, `*.h5`) are ignored by `.gitignore`. Share checkpoints via team drive or release assets.
