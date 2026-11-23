"""Evaluation & Benchmark Script for Autonomous Probe Control Subsystem.

Owner: Maruthi (Module 3)
Evaluates discrete PID controller on 4 disturbance scenarios (Step, Sinusoidal, Drift, Lost)
in both Oracle Mode (GT mask) and Full Closed-Loop Mode (U-Net detect()).

Calculates:
  - Rise time (t_10% -> 90%)
  - Settling time (within +/- 2% band)
  - Maximum overshoot percentage
  - Steady-state error
  - IAE (Integral Absolute Error)
  - ITAE (Integral Time-weighted Absolute Error)
Generates:
  - carotid_project/reports/control_response.png
"""

import os
import sys
import glob
import time
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup module paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pid import DiscretePID
from probe_sim import ProbeKinematicsSimulator
from scene_sim import MotionSceneSimulator
from tracking.tracker import KalmanLumenTracker
from detection.detect import detect


def run_scenario_simulation(
    scenario: str,
    base_img: np.ndarray,
    base_mask: np.ndarray,
    mode: str = "oracle", # "oracle" or "closed_loop"
    kp: float = 0.65,
    ki: float = 0.12,
    kd: float = 0.08,
    dt: float = 0.05,
    sim_frames: int = 80, # 4.0 seconds
) -> Dict[str, Any]:
    """Execute closed-loop simulation on a specific scenario and mode."""
    scene = MotionSceneSimulator(base_img, base_mask, dt=dt)
    pid = DiscretePID(kp=kp, ki=ki, kd=kd, dt=dt, output_limit=50.0, integral_limit=25.0)
    probe = ProbeKinematicsSimulator(dt=dt, max_velocity_mm_s=80.0, actuator_delay_frames=1)
    tracker = KalmanLumenTracker(dt=dt, max_coast_frames=5)

    history = {
        "time": [],
        "disturbance_x": [],
        "probe_pos_x": [],
        "tracking_error_px": [],
        "control_u": [],
        "p_term": [],
        "i_term": [],
        "d_term": [],
        "tracking_status": [],
    }

    curr_probe_x = 0.0

    for k in range(sim_frames):
        t = k * dt

        # 1. Scene updates image based on relative position
        scene_out = scene.step(probe_pos_x=curr_probe_x, scenario=scenario)
        frame = scene_out["frame"]
        gt_mask = scene_out["gt_mask"]
        dist_x = scene_out["dist_x"]

        # 2. Artery detection
        if mode == "oracle":
            mask = gt_mask
        else:
            # Full closed loop: run actual U-Net model on shifted frame
            mask = detect(frame)

        # 3. Tracking & state estimation
        track_state = tracker.update(mask)
        # Tracking error ex = xd - cx (positive if artery moved right relative to probe)
        # Notice: cx_error in camera frame = target_center - cx
        # When probe moves right by +dx, cx shifts left by -dx.
        # Therefore, to drive cx back to 128, error is (cx - 128)
        raw_cx = track_state["cx"]
        err_to_center = float(raw_cx - 128.0) # Error that probe must eliminate

        # 4. PID computation
        u, breakdown = pid.compute(err_to_center)

        # 5. Probe kinematics step
        probe_state = probe.step(cmd_vel_x=u)
        curr_probe_x = probe_state["pos_px_x"]

        # Log
        history["time"].append(t)
        history["disturbance_x"].append(dist_x)
        history["probe_pos_x"].append(curr_probe_x)
        history["tracking_error_px"].append(err_to_center)
        history["control_u"].append(u)
        history["p_term"].append(breakdown["p_term"])
        history["i_term"].append(breakdown["i_term"])
        history["d_term"].append(breakdown["d_term"])
        history["tracking_status"].append(track_state["status"])

    # Metrics calculation
    errs = np.array(history["tracking_error_px"])
    times = np.array(history["time"])
    iae = float(np.sum(np.abs(errs)) * dt)
    itae = float(np.sum(times * np.abs(errs)) * dt)
    steady_err = float(np.mean(np.abs(errs[-15:]))) # Last 15 frames

    # Specific step metrics if scenario is step
    overshoot_pct = 0.0
    settling_time_s = 0.0
    rise_time_s = 0.0

    if scenario == "step":
        step_amp = 25.0
        step_idx = 10
        post_mask = times >= (step_idx * dt)
        post_times = times[post_mask] - (step_idx * dt)
        post_probe = np.array(history["probe_pos_x"])[post_mask]

        max_pos = float(np.max(post_probe))
        overshoot_px = max(0.0, max_pos - step_amp)
        overshoot_pct = (overshoot_px / step_amp) * 100.0

        # Settling time (2% band = +/- 0.5 px)
        band = 0.02 * step_amp
        settling_time_s = float(post_times[-1])
        for idx in range(len(post_probe) - 1, -1, -1):
            if np.abs(post_probe[idx] - step_amp) > band:
                settling_time_s = float(post_times[min(idx + 1, len(post_times) - 1)])
                break

        # Rise time (10% to 90%)
        t10, t90 = None, None
        for idx, pos in enumerate(post_probe):
            if pos >= 0.10 * step_amp and t10 is None:
                t10 = float(post_times[idx])
            if pos >= 0.90 * step_amp and t90 is None:
                t90 = float(post_times[idx])
                break
        rise_time_s = (t90 - t10) if (t10 is not None and t90 is not None) else float(post_times[-1])

    return {
        "scenario": scenario,
        "mode": mode,
        "iae": iae,
        "itae": itae,
        "steady_state_error_px": steady_err,
        "overshoot_pct": overshoot_pct,
        "settling_time_s": settling_time_s,
        "rise_time_s": rise_time_s,
        "history": history,
    }


def evaluate_all_scenarios(
    test_dir: str = "carotid_project/dataset/processed/test",
    output_png: str = "carotid_project/reports/control_response.png",
) -> Dict[str, Any]:
    print("=" * 65)
    print("CONTROL SUBSYSTEM BENCHMARK EVALUATION (Maruthi)")
    print("=" * 65)

    # Pick a representative test frame with clear artery
    sample_imgs = sorted(glob.glob(os.path.join(test_dir, "images", "*.png")))
    sample_masks = sorted(glob.glob(os.path.join(test_dir, "masks", "*.png")))
    assert len(sample_imgs) > 0, "No test images found!"

    # Load 10th test sample
    idx = min(10, len(sample_imgs) - 1)
    base_img = cv2.imread(sample_imgs[idx], cv2.IMREAD_GRAYSCALE)
    base_mask = cv2.imread(sample_masks[idx], cv2.IMREAD_GRAYSCALE)

    scenarios = ["step", "sinusoidal", "drift", "lost"]
    results = {}

    print(f"{'Scenario':<14} | {'Mode':<12} | {'Rise Time':<10} | {'Settling Time':<14} | {'Overshoot':<10} | {'Steady Err':<11} | {'ITAE'}")
    print("-" * 90)

    for sc in scenarios:
        # 1. Oracle mode (GT mask)
        res_oracle = run_scenario_simulation(sc, base_img, base_mask, mode="oracle")
        print(f"{sc:<14} | {'Oracle':<12} | {res_oracle['rise_time_s']:<10.2f}s | {res_oracle['settling_time_s']:<14.2f}s | {res_oracle['overshoot_pct']:<10.1f}% | {res_oracle['steady_state_error_px']:<11.2f}px | {res_oracle['itae']:.2f}")

        # 2. Closed-loop mode (with U-Net detect())
        res_closed = run_scenario_simulation(sc, base_img, base_mask, mode="closed_loop")
        print(f"{sc:<14} | {'Closed-Loop':<12} | {res_closed['rise_time_s']:<10.2f}s | {res_closed['settling_time_s']:<14.2f}s | {res_closed['overshoot_pct']:<10.1f}% | {res_closed['steady_state_error_px']:<11.2f}px | {res_closed['itae']:.2f}")

        results[f"{sc}_oracle"] = res_oracle
        results[f"{sc}_closed_loop"] = res_closed

    print("=" * 65)

    # Plot multi-panel response figures
    plot_control_figures(results, output_png)

    return results


def plot_control_figures(results: Dict[str, Any], save_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    scenarios = ["step", "sinusoidal", "drift", "lost"]

    fig, axes = plt.subplots(4, 2, figsize=(15, 14), facecolor='#121620')

    titles_left = [
        "Step Disturbance: Target vs. Probe Position",
        "Sinusoidal Breathing: Target vs. Probe Position",
        "Continuous Probe Drift: Target vs. Probe Position",
        "Momentary Target Dropout: Target vs. Probe Position"
    ]
    titles_right = [
        "Step: Tracking Error & PID Terms",
        "Sinusoidal: Tracking Error & PID Terms",
        "Drift: Tracking Error & PID Terms",
        "Dropout: Tracking Error & Kalman Coasting"
    ]

    for idx, sc in enumerate(scenarios):
        res = results[f"{sc}_closed_loop"]
        h = res["history"]
        t = h["time"]

        # Left plot: Trajectories
        ax1 = axes[idx, 0]
        ax1.set_facecolor('#1F2634')
        ax1.plot(t, h["disturbance_x"], label="Artery Disturbance", color="#FFAE42", linestyle="--", linewidth=1.8)
        ax1.plot(t, h["probe_pos_x"], label="Robotic Probe Position", color="#24A6FF", linewidth=2.2)
        ax1.set_title(titles_left[idx], color="#F2F6FC", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Position (px)", color="#AEBCCF")
        ax1.set_xlabel("Time (seconds)", color="#AEBCCF")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(facecolor="#1F2634", edgecolor="#384458", labelcolor="#F2F6FC", fontsize=8)
        ax1.tick_params(colors="#AEBCCF")

        # Right plot: Error & PID control terms
        ax2 = axes[idx, 1]
        ax2.set_facecolor('#1F2634')
        ax2.plot(t, h["tracking_error_px"], label="Tracking Error e(t)", color="#38D6E1", linewidth=2.0)
        ax2.plot(t, h["p_term"], label="P-term", color="#34BE70", linestyle=":", linewidth=1.2)
        ax2.plot(t, h["i_term"], label="I-term", color="#FFAE42", linestyle=":", linewidth=1.2)
        ax2.plot(t, h["d_term"], label="D-term", color="#E15B64", linestyle=":", linewidth=1.2)
        ax2.set_title(titles_right[idx], color="#F2F6FC", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Signal Amplitude", color="#AEBCCF")
        ax2.set_xlabel("Time (seconds)", color="#AEBCCF")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(facecolor="#1F2634", edgecolor="#384458", labelcolor="#F2F6FC", fontsize=8)
        ax2.tick_params(colors="#AEBCCF")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Control response plots saved to: {save_path}")


if __name__ == "__main__":
    evaluate_all_scenarios()
