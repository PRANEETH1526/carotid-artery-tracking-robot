"""Vessel Tracking & State Estimation Module.

Owner: Shabrish
Component: Real-Time Centroid Tracking & Kalman Filter State Estimation.

Computes artery lumen centroid, area, equivalent diameter, and tracking error
from binary segmentation masks. Integrates a discrete Kalman filter with
constant-velocity kinematics for missing-frame prediction and outlier rejection.
"""

from enum import Enum
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


class TrackingStatus(str, Enum):
    TRACKING = "TRACKING"      # Active valid detection with Kalman measurement update
    PREDICTED = "PREDICTED"    # Detection dropped or outlier; coasting on Kalman prediction
    LOST = "LOST"              # No valid target for > max_coast_frames


def compute_mask_properties(
    mask: np.ndarray,
    target_center: Tuple[float, float] = (128.0, 128.0),
    px_per_mm: float = 1.0, # Assumed scale (no physical scanner calibration available)
) -> Optional[Dict[str, Any]]:
    """Compute centroid, area, equivalent diameter, and tracking errors from a binary mask.

    Args:
        mask: 2D uint8 binary mask (256x256), strictly {0, 255}.
        target_center: Assumed desired center (xd, yd) in pixels. Default: (128.0, 128.0).
        px_per_mm: Conversion factor. Default 1.0 (labeled as ASSUMED SCALE).

    Returns:
        Dict with geometrical properties, or None if mask is empty.
    """
    if mask is None or np.count_nonzero(mask == 255) == 0:
        return None

    # Calculate spatial moments
    M = cv2.moments(mask)
    if M["m00"] <= 0:
        return None

    # Centroid in pixels
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    # Area in pixels (mask values are 255, so M00 / 255 gives pixel count)
    area_px = float(M["m00"] / 255.0)

    # Equivalent circular diameter: d = 2 * sqrt(Area / pi)
    eq_diameter_px = 2.0 * np.sqrt(area_px / np.pi)

    # Tracking error e = xd - x (positive error means artery is to the left/top of target)
    xd, yd = target_center
    ex_px = float(xd - cx)
    ey_px = float(yd - cy)

    # Real-world units using assumed scale
    ex_mm = ex_px / px_per_mm
    ey_mm = ey_px / px_per_mm

    return {
        "cx": cx,
        "cy": cy,
        "area_px": area_px,
        "diameter_px": eq_diameter_px,
        "ex_px": ex_px,
        "ey_px": ey_px,
        "ex_mm_assumed": ex_mm,
        "ey_mm_assumed": ey_mm,
    }


class KalmanLumenTracker:
    """Constant-velocity discrete Kalman filter for carotid artery tracking.

    State vector: x = [cx, cy, vx, vy]^T
    Measurement:  z = [cx, cy]^T
    """

    def __init__(
        self,
        dt: float = 0.05,            # Frame interval (0.05s = 20 FPS)
        target_center: Tuple[float, float] = (128.0, 128.0),
        px_per_mm: float = 1.0,      # Assumed scale (1 px = 1 unit)
        max_coast_frames: int = 5,   # Maximum frames to predict before declaring LOST
        max_jump_px: float = 60.0,   # Maximum plausible frame-to-frame centroid displacement
        min_area_px: float = 30.0,   # Reject smaller blobs as noise
        max_area_px: float = 8000.0, # Reject gigantic blobs as ultrasound artifacts
        process_noise_std: float = 2.0,
        measurement_noise_std: float = 1.5,
    ):
        self.dt = dt
        self.target_center = target_center
        self.px_per_mm = px_per_mm
        self.max_coast_frames = max_coast_frames
        self.max_jump_px = max_jump_px
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px

        # State vector [cx, cy, vx, vy]^T
        self.x = np.zeros(4, dtype=np.float64)
        # Covariance matrix P
        self.P = np.eye(4, dtype=np.float64) * 100.0

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance Q
        q = process_noise_std ** 2
        dt2 = 0.5 * (dt ** 2)
        dt3 = dt ** 3 / 3.0
        dt4 = dt ** 4 / 4.0
        self.Q = q * np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0,  dt, 0.0],
            [0.0, dt3, 0.0,  dt]
        ], dtype=np.float64)

        # Measurement noise covariance R
        r = measurement_noise_std ** 2
        self.R = np.array([
            [r, 0.0],
            [0.0, r]
        ], dtype=np.float64)

        self.initialized = False
        self.coast_count = 0
        self.last_valid_area = 0.0
        self.status = TrackingStatus.LOST

    def predict(self) -> Tuple[float, float]:
        """Kalman time update / state extrapolation."""
        if not self.initialized:
            return self.target_center

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, mask: Optional[np.ndarray]) -> Dict[str, Any]:
        """Process new segmentation mask and return current state estimate."""
        # Step 1: Kalman Predict step
        pred_cx, pred_cy = self.predict()

        # Step 2: Extract visual measurement
        meas = compute_mask_properties(mask, self.target_center, self.px_per_mm) if mask is not None else None

        valid_measurement = False
        area = 0.0
        diameter = 0.0

        if meas is not None:
            raw_cx = meas["cx"]
            raw_cy = meas["cy"]
            area = meas["area_px"]
            diameter = meas["diameter_px"]

            # Outlier checks: area thresholds and maximum jump distance
            area_valid = (self.min_area_px <= area <= self.max_area_px)
            if self.initialized:
                dist = np.hypot(raw_cx - self.x[0], raw_cy - self.x[1])
                jump_valid = (dist <= self.max_jump_px)
            else:
                jump_valid = True

            if area_valid and jump_valid:
                valid_measurement = True

        # Step 3: Kalman Measurement update or Coast
        if valid_measurement:
            z = np.array([raw_cx, raw_cy], dtype=np.float64)

            if not self.initialized:
                # Initialize state
                self.x = np.array([raw_cx, raw_cy, 0.0, 0.0], dtype=np.float64)
                self.P = np.eye(4, dtype=np.float64) * 5.0
                self.initialized = True
            else:
                # Standard Kalman Innovation & Kalman Gain update
                y = z - (self.H @ self.x)
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                I = np.eye(4, dtype=np.float64)
                self.P = (I - K @ self.H) @ self.P

            self.coast_count = 0
            self.last_valid_area = area
            self.status = TrackingStatus.TRACKING
        else:
            # Measurement missing or rejected
            if self.initialized:
                self.coast_count += 1
                if self.coast_count <= self.max_coast_frames:
                    self.status = TrackingStatus.PREDICTED
                else:
                    self.status = TrackingStatus.LOST
            else:
                self.status = TrackingStatus.LOST

        est_cx = float(self.x[0]) if self.initialized else float(self.target_center[0])
        est_cy = float(self.x[1]) if self.initialized else float(self.target_center[1])
        vx = float(self.x[2]) if self.initialized else 0.0
        vy = float(self.x[3]) if self.initialized else 0.0

        xd, yd = self.target_center
        ex = xd - est_cx
        ey = yd - est_cy

        return {
            "cx": est_cx,
            "cy": est_cy,
            "vx": vx,
            "vy": vy,
            "ex_px": ex,
            "ey_px": ey,
            "ex_mm_assumed": ex / self.px_per_mm,
            "ey_mm_assumed": ey / self.px_per_mm,
            "status": self.status.value,
            "area_px": area if valid_measurement else self.last_valid_area,
            "diameter_px": diameter,
            "is_tracking": (self.status == TrackingStatus.TRACKING),
        }

    def reset(self):
        """Reset tracker state."""
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.initialized = False
        self.coast_count = 0
        self.last_valid_area = 0.0
        self.status = TrackingStatus.LOST
