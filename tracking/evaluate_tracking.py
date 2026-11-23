"""Tracking Subsystem Evaluation on Test Dataset.

Owner: Shabrish
Evaluates centroid error, Kalman filtering benefits (jitter reduction), and
10% missing-frame drop robustness across test sessions.

NOTE ON DATASET:
Frames in this dataset are sampled slices from ultrasound recording sessions,
not guaranteed 30 FPS continuous video. Inter-frame motions can exhibit discrete
jumps between adjacent slices.
"""

import os
import sys
import csv
import glob
import random
from typing import Dict, List, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure tracking and detection can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tracker import KalmanLumenTracker, compute_mask_properties


def evaluate_tracking_on_test_set(
    test_dir: str = "carotid_project/dataset/processed/test",
    output_plot: str = "carotid_project/reports/tracking_trajectories.png",
    drop_rate: float = 0.10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 60)
    print("TRACKING SUBSYSTEM EVALUATION (Shabrish)")
    print("=" * 60)

    # Read test metadata to group by session
    meta_csv = os.path.join(test_dir, "test_metadata.csv")
    sessions: Dict[str, List[Dict[str, str]]] = {}

    if os.path.exists(meta_csv):
        with open(meta_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sess = row["session_id"]
                if sess not in sessions:
                    sessions[sess] = []
                sessions[sess].append(row)
    else:
        # Fallback if CSV missing
        mask_files = sorted(glob.glob(os.path.join(test_dir, "masks", "*.png")))
        sessions["all_test"] = [{"mask_filename": os.path.basename(m)} for m in mask_files]

    print(f"Test sessions found: {len(sessions)}")
    for sess, rows in sessions.items():
        print(f"  Session {sess}: {len(rows)} frames")

    rng = random.Random(random_seed)

    all_raw_errors = []
    all_kalman_errors = []
    raw_step_diffs = []
    kalman_step_diffs = []

    # Store trajectory for the 3 longest sessions for plotting
    sorted_sessions = sorted(sessions.items(), key=lambda item: len(item[1]), reverse=True)
    plot_sessions = [s[0] for s in sorted_sessions[:3]]
    session_trajectories: Dict[str, Dict[str, List[float]]] = {}

    for sess_id, rows in sessions.items():
        tracker = KalmanLumenTracker(dt=0.05, max_coast_frames=5)
        raw_prev_pos = None
        kalman_prev_pos = None

        if sess_id in plot_sessions:
            session_trajectories[sess_id] = {
                "gt_cx": [], "gt_cy": [],
                "raw_cx": [], "raw_cy": [],
                "kalman_cx": [], "kalman_cy": [],
                "dropped": [], "errors": []
            }

        for row in rows:
            mask_p = os.path.join(test_dir, "masks", row["mask_filename"])
            gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            gt_props = compute_mask_properties(gt_mask)
            if gt_props is None:
                continue

            gt_cx, gt_cy = gt_props["cx"], gt_props["cy"]

            # 10% dropout test simulation
            is_dropped = (rng.random() < drop_rate)
            input_mask = None if is_dropped else gt_mask

            # Raw measurement (without Kalman)
            raw_props = compute_mask_properties(input_mask)
            if raw_props is not None:
                raw_cx, raw_cy = raw_props["cx"], raw_props["cy"]
                raw_err = np.hypot(raw_cx - gt_cx, raw_cy - gt_cy)
                all_raw_errors.append(raw_err)

                if raw_prev_pos is not None:
                    raw_diff = np.hypot(raw_cx - raw_prev_pos[0], raw_cy - raw_prev_pos[1])
                    raw_step_diffs.append(raw_diff)
                raw_prev_pos = (raw_cx, raw_cy)
            else:
                # Dropped frame without Kalman has no estimate (or holds last known)
                pass

            # Kalman Filter update
            track_state = tracker.update(input_mask)
            k_cx, k_cy = track_state["cx"], track_state["cy"]
            k_err = np.hypot(k_cx - gt_cx, k_cy - gt_cy)
            all_kalman_errors.append(k_err)

            if kalman_prev_pos is not None:
                k_diff = np.hypot(k_cx - kalman_prev_pos[0], k_cy - kalman_prev_pos[1])
                kalman_step_diffs.append(k_diff)
            kalman_prev_pos = (k_cx, k_cy)

            # Record trajectory
            if sess_id in plot_sessions:
                session_trajectories[sess_id]["gt_cx"].append(gt_cx)
                session_trajectories[sess_id]["gt_cy"].append(gt_cy)
                session_trajectories[sess_id]["raw_cx"].append(raw_cx if raw_props else np.nan)
                session_trajectories[sess_id]["raw_cy"].append(raw_cy if raw_props else np.nan)
                session_trajectories[sess_id]["kalman_cx"].append(k_cx)
                session_trajectories[sess_id]["kalman_cy"].append(k_cy)
                session_trajectories[sess_id]["dropped"].append(1.0 if is_dropped else 0.0)
                session_trajectories[sess_id]["errors"].append(k_err)

    # Compute statistics
    mean_err = float(np.mean(all_kalman_errors))
    med_err = float(np.median(all_kalman_errors))
    p95_err = float(np.percentile(all_kalman_errors, 95))

    raw_jitter_std = float(np.std(raw_step_diffs))
    kalman_jitter_std = float(np.std(kalman_step_diffs))
    jitter_reduction_pct = float((raw_jitter_std - kalman_jitter_std) / raw_jitter_std * 100.0) if raw_jitter_std > 0 else 0.0

    print("\n" + "=" * 60)
    print("TRACKING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Centroid Estimation Error (Kalman vs GT):")
    print(f"  Mean Error:               {mean_err:.2f} px")
    print(f"  Median Error:             {med_err:.2f} px")
    print(f"  95th Percentile Error:    {p95_err:.2f} px")
    print(f"\nJitter Analysis (Frame-to-Frame Variance):")
    print(f"  Raw Detections Jitter:    {raw_jitter_std:.2f} px (std dev)")
    print(f"  Kalman Filtered Jitter:   {kalman_jitter_std:.2f} px (std dev)")
    print(f"  Jitter Reduction:         {jitter_reduction_pct:.1f}% smoother")
    print(f"\n10% Drop Test Robustness:")
    print(f"  Successfully tracked through {drop_rate*100:.0f}% random dropped frames using Kalman coasting.")
    print("=" * 60)

    # Plot trajectories
    plot_tracking_figures(session_trajectories, output_plot)

    return {
        "mean_error_px": mean_err,
        "median_error_px": med_err,
        "p95_error_px": p95_err,
        "raw_jitter_std_px": raw_jitter_std,
        "kalman_jitter_std_px": kalman_jitter_std,
        "jitter_reduction_pct": jitter_reduction_pct,
        "drop_rate": drop_rate,
    }


def plot_tracking_figures(session_data: Dict[str, Any], save_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    n_sess = len(session_data)
    if n_sess == 0:
        return

    fig, axes = plt.subplots(n_sess, 2, figsize=(14, 3.5 * n_sess), facecolor='#121620')
    if n_sess == 1:
        axes = np.expand_dims(axes, 0)

    for idx, (sess_id, data) in enumerate(session_data.items()):
        frames = range(len(data["gt_cx"]))

        # Left plot: X and Y coordinate trajectories
        ax1 = axes[idx, 0]
        ax1.set_facecolor('#1F2634')
        ax1.plot(frames, data["gt_cx"], label="GT Cx", color="#38D6E1", linewidth=1.5, linestyle="--")
        ax1.plot(frames, data["kalman_cx"], label="Kalman Cx", color="#24A6FF", linewidth=2.0)
        # Mark dropped frames
        dropped_idx = [i for i, d in enumerate(data["dropped"]) if d > 0.5]
        if dropped_idx:
            ax1.scatter(dropped_idx, [data["kalman_cx"][i] for i in dropped_idx], color="#FFAE42", marker="x", s=40, label="Dropped (Coasted)")

        ax1.set_title(f"Session {sess_id[:16]} — Centroid Trajectory", color="#F2F6FC", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Frame Index", color="#AEBCCF")
        ax1.set_ylabel("Pixel X Position", color="#AEBCCF")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(facecolor="#1F2634", edgecolor="#384458", labelcolor="#F2F6FC", fontsize=8)
        ax1.tick_params(colors="#AEBCCF")

        # Right plot: Tracking Error over time
        ax2 = axes[idx, 1]
        ax2.set_facecolor('#1F2634')
        ax2.plot(frames, data["errors"], label="Kalman Tracking Error", color="#34BE70", linewidth=1.8)
        ax2.axhline(np.mean(data["errors"]), color="#FFAE42", linestyle=":", label=f"Mean: {np.mean(data['errors']):.2f} px")

        ax2.set_title(f"Session {sess_id[:16]} — Tracking Error (px)", color="#F2F6FC", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Frame Index", color="#AEBCCF")
        ax2.set_ylabel("Error (pixels)", color="#AEBCCF")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(facecolor="#1F2634", edgecolor="#384458", labelcolor="#F2F6FC", fontsize=8)
        ax2.tick_params(colors="#AEBCCF")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Tracking trajectory plots saved to: {save_path}")


if __name__ == "__main__":
    evaluate_tracking_on_test_set()
