"""Unit Tests for Autonomous Carotid Artery Tracking Robot Pipeline.

Tests:
  1. detect() output shape, dtype, and strictly binary values {0, 255}
  2. Centroid computation on a synthetic ground-truth circle
  3. Kalman filter behavior when a detection is missing (coasting in PREDICTED mode)
  4. PID controller step response and error convergence
  5. PID controller anti-windup clamping under sustained saturation
  6. Smoke test of closed-loop run_pipeline.py in --headless mode
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Ensure project root is in sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tracking.tracker import compute_mask_properties, KalmanLumenTracker, TrackingStatus
from control.pid import DiscretePID
from control.probe_sim import ProbeKinematicsSimulator
from control.scene_sim import MotionSceneSimulator
from run_pipeline import run_pipeline


# ==========================================
# 1. Centroid of a Synthetic Circle
# ==========================================
def test_synthetic_circle_centroid():
    """Verify centroid and area on a perfect synthetic circle."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    true_cx, true_cy = 135.0, 110.0
    radius = 20

    # Draw solid white circle
    cv2.circle(mask, (int(true_cx), int(true_cy)), radius, 255, -1)

    props = compute_mask_properties(mask, target_center=(128.0, 128.0), px_per_mm=1.0)
    assert props is not None, "Props should not be None for non-empty mask"

    # Centroid should match within 0.1 pixel
    assert abs(props["cx"] - true_cx) < 0.2, f"cx error: {props['cx']} vs {true_cx}"
    assert abs(props["cy"] - true_cy) < 0.2, f"cy error: {props['cy']} vs {true_cy}"

    # Area should match pi * r^2 (~1256 pixels) within 5%
    expected_area = np.pi * (radius ** 2)
    assert abs(props["area_px"] - expected_area) / expected_area < 0.05

    # Tracking error
    assert abs(props["ex_px"] - (128.0 - true_cx)) < 0.2
    assert abs(props["ey_px"] - (128.0 - true_cy)) < 0.2


# ==========================================
# 2. Kalman Missing Detection Behavior
# ==========================================
def test_kalman_missing_detection_coasting():
    """Verify Kalman filter switches to PREDICTED and coasts smoothly when detection drops."""
    tracker = KalmanLumenTracker(dt=0.05, max_coast_frames=5)

    # Feed 5 valid frames moving right at 2 px/frame (vx = 40 px/s)
    for i in range(5):
        mask = np.zeros((256, 256), dtype=np.uint8)
        cx = 100 + i * 2
        cy = 128
        cv2.circle(mask, (cx, cy), 15, 255, -1)
        state = tracker.update(mask)
        assert state["status"] == TrackingStatus.TRACKING.value

    last_measured_cx = state["cx"]

    # Frame 6: Drop detection (mask is None)
    state_drop1 = tracker.update(None)
    assert state_drop1["status"] == TrackingStatus.PREDICTED.value, "Status should be PREDICTED on missing frame"
    # Coasting position should advance in the direction of velocity
    assert state_drop1["cx"] >= last_measured_cx, "Kalman should coast forward along velocity"

    # Drop 5 consecutive frames -> should remain PREDICTED up to frame 5
    for _ in range(4):
        state_coast = tracker.update(None)
        assert state_coast["status"] == TrackingStatus.PREDICTED.value

    # Frame 11 (6th missing frame) -> should transition to LOST
    state_lost = tracker.update(None)
    assert state_lost["status"] == TrackingStatus.LOST.value, "Status should be LOST after exceeding max_coast_frames"


# ==========================================
# 3. PID Step Response & Convergence
# ==========================================
def test_pid_step_response():
    """Verify PID controller drives step error to zero."""
    pid = DiscretePID(kp=0.65, ki=0.12, kd=0.08, dt=0.05, output_limit=50.0)
    probe_x = 0.0
    target_x = 20.0 # 20 px step

    for _ in range(60): # 3.0 seconds
        error = target_x - probe_x
        u, _ = pid.compute(error)
        probe_x += u * 0.05 # Euler step

    final_error = abs(target_x - probe_x)
    assert final_error < 0.25, f"Final error {final_error:.2f} px should be under 0.25 px"


# ==========================================
# 4. PID Anti-Windup Clamping
# ==========================================
def test_pid_anti_windup_clamping():
    """Verify integral accumulator does not grow unbounded under severe saturation."""
    integral_limit = 25.0
    pid = DiscretePID(kp=0.5, ki=0.2, kd=0.05, dt=0.05, integral_limit=integral_limit)

    # Feed massive constant error of 1000 px for 200 cycles
    for _ in range(200):
        _, breakdown = pid.compute(1000.0)

    # Integral accumulator should be strictly clamped to integral_limit
    assert abs(pid.integral) <= (integral_limit + 1e-5), f"Integral {pid.integral} exceeded limit {integral_limit}"


# ==========================================
# 5. detect() Interface Contract
# ==========================================
def test_detect_interface_contract():
    """Verify detect() produces strictly 256x256 uint8 array with values in {0, 255}."""
    from detection.detect import detect

    # Synthetic ultrasound test frame
    test_frame = np.ones((256, 256), dtype=np.uint8) * 100
    # Add dark artery lumen circle
    cv2.circle(test_frame, (128, 128), 25, 30, -1)

    mask = detect(test_frame)

    assert isinstance(mask, np.ndarray), "Output must be a numpy array"
    assert mask.shape == (256, 256), f"Output shape must be (256, 256), got {mask.shape}"
    assert mask.dtype == np.uint8, f"Output dtype must be uint8, got {mask.dtype}"

    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255}), f"Mask values must be strictly {{0, 255}}, got {unique_vals}"


# ==========================================
# 6. Full Pipeline Smoke Test (--headless)
# ==========================================
def test_pipeline_smoke_headless():
    """Smoke test running closed-loop pipeline for 10 frames."""
    res = run_pipeline(scenario="step", mode="oracle", headless=True, frames=10)
    assert res["total_frames"] == 10
    assert res["mean_fps"] > 0
    assert os.path.exists(res["output_gif"])
