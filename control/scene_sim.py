"""Ultrasound Scene & Motion Disturbance Simulator.

Owner: Maruthi (Module 3)
Component: Autonomous Probe Motion Control

Simulates relative motion between carotid artery and robotic probe by shifting
real ultrasound frames and masks in real-time.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class MotionSceneSimulator:
    """Simulates carotid artery motion disturbances and probe visual servoing feedback."""

    def __init__(
        self,
        base_image: np.ndarray,
        base_mask: np.ndarray,
        dt: float = 0.05,
        target_center: Tuple[float, float] = (128.0, 128.0),
    ):
        assert base_image.shape == (256, 256), "Base image must be 256x256"
        assert base_mask.shape == (256, 256), "Base mask must be 256x256"

        self.base_image = base_image.copy()
        self.base_mask = base_mask.copy()
        self.dt = dt
        self.target_center = target_center

        self.frame_index = 0
        self.current_time = 0.0
        self.drift_accum_x = 0.0
        self.drift_accum_y = 0.0

    def get_disturbance(self, scenario: str) -> Tuple[float, float, bool]:
        """Compute external artery disturbance offset (dx, dy) in pixels and target visibility.

        Returns:
            (dist_x, dist_y, is_visible)
        """
        t = self.current_time
        k = self.frame_index
        dist_x = 0.0
        dist_y = 0.0
        is_visible = True

        if scenario == "step":
            # 25-pixel step offset after frame 10
            if k >= 10:
                dist_x = 25.0
                dist_y = 0.0

        elif scenario == "sinusoidal":
            # 18-pixel sinusoidal breathing motion at 0.25 Hz (15 breaths/minute)
            dist_x = 18.0 * np.sin(2.0 * np.pi * 0.25 * t)
            dist_y = 4.0 * np.cos(2.0 * np.pi * 0.25 * t)

        elif scenario == "drift":
            # Slow continuous drift (2 px/s) + small random walk jitter
            self.drift_accum_x += 2.0 * self.dt + np.random.normal(0, 0.4)
            self.drift_accum_y += 0.5 * self.dt + np.random.normal(0, 0.2)
            dist_x = self.drift_accum_x
            dist_y = self.drift_accum_y

        elif scenario == "lost":
            # Step offset of 20 px, then artery disappears for 10 frames (acoustic dropout)
            if k >= 10:
                dist_x = 20.0
            if 25 <= k < 35:
                is_visible = False

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return float(dist_x), float(dist_y), is_visible

    def step(
        self,
        probe_pos_x: float,
        probe_pos_y: float = 0.0,
        scenario: str = "step",
    ) -> Dict[str, Any]:
        """Shift image and mask by relative displacement (artery disturbance - probe position).

        Returns:
            Dict containing shifted frame, shifted GT mask, and disturbance info.
        """
        dist_x, dist_y, is_visible = self.get_disturbance(scenario)

        # Net apparent shift in camera frame = artery motion - probe motion
        shift_x = dist_x - probe_pos_x
        shift_y = dist_y - probe_pos_y

        # Translation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Warp ultrasound image (with reflection border fill for realistic acoustic look)
        shifted_img = cv2.warpAffine(
            self.base_image, M, (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Warp ground-truth mask (with zero border fill)
        if is_visible:
            shifted_mask = cv2.warpAffine(
                self.base_mask, M, (256, 256),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            # Artery temporarily occluded or out-of-plane
            shifted_mask = np.zeros_like(self.base_mask)

        self.frame_index += 1
        self.current_time += self.dt

        return {
            "frame": shifted_img,
            "gt_mask": shifted_mask,
            "dist_x": dist_x,
            "dist_y": dist_y,
            "shift_x": shift_x,
            "shift_y": shift_y,
            "is_visible": is_visible,
            "frame_idx": self.frame_index,
            "time_s": self.current_time,
        }

    def reset(self):
        """Reset scene state."""
        self.frame_index = 0
        self.current_time = 0.0
        self.drift_accum_x = 0.0
        self.drift_accum_y = 0.0
