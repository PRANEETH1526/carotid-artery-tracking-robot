# Vessel Tracking & State Estimation Module

**Owner:** Shabrish  
**Component:** Real-Time Centroid Tracking & Error Estimation

---

## Purpose
Track the carotid artery lumen center across consecutive ultrasound video frames, calculate the offset (error) from the desired robotic probe center, and provide state estimation (velocity, confidence, and lost-target detection).

---

## Interface Contract

### Input
- **Binary Mask:** 2D numpy array `mask` of shape `(256, 256)` and dtype `uint8` (`0` or `255`), received from Srujan's `detection.detect()`.

### Output
- **Target Centroid:** $(x, y)$ coordinate in pixels (or millimeters) of the lumen center.
- **Tracking Error:** $e_x = x_d - x$, $e_y = y_d - y$, where $(x_d, y_d)$ is probe target center (default: $(128, 128)$ pixels).
- **Status Flag:** Tracking state boolean (`is_tracking: bool`, e.g., `False` if lumen is lost or occluded).
- **Calibration:** Pixel-to-millimeter scaling factor (e.g., based on ultrasound depth scale).

### Standard Function Signature
```python
from typing import Tuple, Dict, Any
import numpy as np

def track_lumen(
    mask: np.ndarray,
    target_center: Tuple[int, int] = (128, 128),
    pixel_to_mm: float = 0.15,
) -> Dict[str, Any]:
    """Calculate centroid and tracking error from a segmented lumen mask.

    Args:
        mask: Binary mask (256x256 uint8, {0, 255}).
        target_center: Desired probe center (x_d, y_d) in pixels. Default (128, 128).
        pixel_to_mm: Conversion factor from pixels to real-world millimeters.

    Returns:
        Dict with keys:
            - 'centroid_px': (cx, cy) in pixels
            - 'error_px': (ex, ey) in pixels
            - 'error_mm': (ex_mm, ey_mm) in millimeters
            - 'is_tracking': bool (True if valid lumen detected, False if lost)
            - 'area_px': total lumen pixel area
    """
    pass
```
