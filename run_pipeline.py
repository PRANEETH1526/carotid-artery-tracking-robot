"""Closed-Loop Integration Pipeline for Autonomous Carotid Artery Tracking Robot.

Connects:
  Live Stream / Scene Simulator -> U-Net Detection -> Kalman Tracking -> PID Control -> Motor Update -> Real-Time HUD

Usage:
  python run_pipeline.py --headless --scenario step --mode model
  python run_pipeline.py --demo --scenario sinusoidal --mode model
"""

import os
import sys
import glob
import time
import argparse
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

# Setup module import roots
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from detection.detect import detect
from tracking.tracker import KalmanLumenTracker
from control.pid import DiscretePID
from control.probe_sim import ProbeKinematicsSimulator
from control.scene_sim import MotionSceneSimulator


def create_hud_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    gt_mask: np.ndarray,
    track_state: Dict[str, Any],
    probe_state: Dict[str, Any],
    control_u: float,
    scenario: str,
    mode: str,
    frame_idx: int,
    fps: float,
    target_center: tuple = (128, 128),
) -> np.ndarray:
    """Render comprehensive telemetry HUD overlay onto ultrasound frame."""
    # Scale from 256x256 to 512x512 for clear HUD rendering
    canvas = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    scale = 2.0 # Coordinate scale factor 256 -> 512

    # 1. Target center crosshair (Cyan dashed)
    tx, ty = int(target_center[0] * scale), int(target_center[1] * scale)
    cv2.drawMarker(canvas, (tx, ty), (56, 214, 225), cv2.MARKER_CROSS, 20, 1)
    cv2.circle(canvas, (tx, ty), 8, (56, 214, 225), 1)

    # 2. Artery contour
    if mask is not None and np.count_nonzero(mask == 255) > 0:
        mask_scaled = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (52, 190, 112), 2) # Green contour

    # 3. Tracked centroid
    cx, cy = track_state["cx"], track_state["cy"]
    cx_s, cy_s = int(cx * scale), int(cy * scale)
    status = track_state["status"]

    if status == "TRACKING":
        marker_color = (255, 174, 66) # Orange: Active tracking
    elif status == "PREDICTED":
        marker_color = (0, 255, 255)   # Yellow: Kalman coasting
    else:
        marker_color = (0, 0, 255)     # Red: Target Lost

    cv2.drawMarker(canvas, (cx_s, cy_s), marker_color, cv2.MARKER_CROSS, 16, 2)

    # Draw error vector from target to centroid
    cv2.line(canvas, (tx, ty), (cx_s, cy_s), (0, 165, 255), 1, cv2.LINE_AA)

    # 4. Top Telemetry Header Bar
    cv2.rectangle(canvas, (0, 0), (512, 60), (18, 22, 32), -1)
    cv2.line(canvas, (0, 60), (512, 60), (56, 214, 225), 1)

    cv2.putText(canvas, f"FRAME: {frame_idx:03d} | SCENARIO: {scenario.upper()} ({mode})", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (242, 246, 252), 1, cv2.LINE_AA)

    status_color = (52, 190, 112) if status == "TRACKING" else ((255, 174, 66) if status == "PREDICTED" else (0, 0, 255))
    cv2.putText(canvas, f"STATUS: {status}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"FPS: {fps:.1f}", (420, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (174, 188, 207), 1, cv2.LINE_AA)

    # 5. Bottom Control State Bar
    cv2.rectangle(canvas, (0, 442), (512, 512), (18, 22, 32), -1)
    cv2.line(canvas, (0, 442), (512, 442), (56, 214, 225), 1)

    ex = track_state["ex_px"]
    ey = track_state["ey_px"]
    probe_x = probe_state["pos_px_x"]
    theta_deg = probe_state["theta_deg_x"]

    cv2.putText(canvas, f"Error (ex, ey): ({ex:+5.1f}, {ey:+5.1f}) px", (12, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (242, 246, 252), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Control u(t): {control_u:+5.1f} px/s | Probe Pos: {probe_x:+5.1f} px", (12, 488),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 214, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Joint Angle theta: {theta_deg:+5.2f} deg", (12, 506),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (174, 188, 207), 1, cv2.LINE_AA)

    return canvas


def run_pipeline(
    scenario: str = "step",
    mode: str = "model", # "oracle" or "model"
    headless: bool = True,
    demo: bool = False,
    frames: int = 80,
    output_gif: str = "carotid_project/results/pipeline_demo.gif",
    test_dir: str = "carotid_project/dataset/processed/test",
):
    print("=" * 65)
    print("AUTONOMOUS CAROTID TRACKING: CLOSED-LOOP PIPELINE")
    print("=" * 65)
    print(f"Scenario:               {scenario}")
    print(f"Mode:                   {mode}")
    print(f"Headless / Save GIF:    {headless}")
    print(f"Frames to run:          {frames}")

    # Resolve paths if called from either workspace root or carotid_project root
    if not os.path.exists(test_dir):
        alt_test_dir = os.path.join(current_dir, "dataset", "processed", "test")
        if os.path.exists(alt_test_dir):
            test_dir = alt_test_dir

    if not os.path.isabs(output_gif):
        if output_gif.startswith("carotid_project/") and not os.path.exists("carotid_project"):
            output_gif = os.path.join(current_dir, output_gif[len("carotid_project/"):])

    # Load representative base test frame
    test_imgs = sorted(glob.glob(os.path.join(test_dir, "images", "*.png")))
    test_masks = sorted(glob.glob(os.path.join(test_dir, "masks", "*.png")))
    assert len(test_imgs) > 0, "No test images found!"

    base_idx = min(10, len(test_imgs) - 1)
    base_img = cv2.imread(test_imgs[base_idx], cv2.IMREAD_GRAYSCALE)
    base_mask = cv2.imread(test_masks[base_idx], cv2.IMREAD_GRAYSCALE)

    dt = 0.05 # 20 Hz
    scene = MotionSceneSimulator(base_img, base_mask, dt=dt)
    tracker = KalmanLumenTracker(dt=dt, max_coast_frames=5)
    pid = DiscretePID(kp=0.65, ki=0.12, kd=0.08, dt=dt, output_limit=50.0, integral_limit=25.0)
    probe = ProbeKinematicsSimulator(dt=dt, max_velocity_mm_s=80.0, actuator_delay_frames=1)

    hud_frames = []
    curr_probe_x = 0.0

    t_pipeline_start = time.time()
    loop_times = []

    for k in range(frames):
        t_loop_0 = time.perf_counter()

        # 1. Scene step (shifts image based on relative position)
        scene_out = scene.step(probe_pos_x=curr_probe_x, scenario=scenario)
        frame = scene_out["frame"]
        gt_mask = scene_out["gt_mask"]

        # 2. Artery detection
        if mode == "oracle":
            mask = gt_mask
        else:
            mask = detect(frame)

        # 3. Tracking & state estimation
        track_state = tracker.update(mask)

        # 4. PID error calculation
        raw_cx = track_state["cx"]
        err_to_center = float(raw_cx - 128.0)

        # 5. PID control
        u, breakdown = pid.compute(err_to_center)

        # 6. Actuator step
        probe_state = probe.step(cmd_vel_x=u)
        curr_probe_x = probe_state["pos_px_x"]

        t_loop_1 = time.perf_counter()
        loop_ms = (t_loop_1 - t_loop_0) * 1000.0
        loop_times.append(loop_ms)
        instant_fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0

        # 7. Render HUD
        hud = create_hud_frame(
            frame=frame,
            mask=mask,
            gt_mask=gt_mask,
            track_state=track_state,
            probe_state=probe_state,
            control_u=u,
            scenario=scenario,
            mode=mode,
            frame_idx=k + 1,
            fps=instant_fps,
        )

        hud_frames.append(hud)

        if demo and not headless:
            cv2.imshow("Autonomous Carotid Artery Tracking Robot HUD", hud)
            if cv2.waitKey(int(dt * 1000)) & 0xFF == 27: # ESC to stop
                break

    if demo and not headless:
        cv2.destroyAllWindows()

    mean_loop_ms = np.mean(loop_times)
    mean_fps = 1000.0 / mean_loop_ms if mean_loop_ms > 0 else 0.0

    print(f"\nPipeline Execution Complete:")
    print(f"  Frames processed:     {len(hud_frames)}")
    print(f"  Mean latency/frame:   {mean_loop_ms:.2f} ms")
    print(f"  Throughput:           {mean_fps:.1f} FPS (Target >= 20 FPS: {'PASSED' if mean_fps >= 20 else 'FAILED'})")

    # Save animated GIF if pillow is available
    if headless and hud_frames:
        os.makedirs(os.path.dirname(os.path.abspath(output_gif)), exist_ok=True)
        try:
            from PIL import Image
            # Downsample frames to 256x256 for compact GIF size
            pil_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (320, 320)), cv2.COLOR_BGR2RGB)) for f in hud_frames[::2]]
            pil_frames[0].save(
                output_gif,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(dt * 2 * 1000),
                loop=0,
                optimize=True,
            )
            print(f"Demo animation successfully saved to: {output_gif}")
        except Exception as e:
            print(f"Could not save GIF via PIL: {e}")

    return {
        "mean_latency_ms": float(mean_loop_ms),
        "mean_fps": float(mean_fps),
        "total_frames": len(hud_frames),
        "output_gif": output_gif,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full carotid tracking closed-loop pipeline.")
    parser.add_argument("--scenario", type=str, default="step", choices=["step", "sinusoidal", "drift", "lost"])
    parser.add_argument("--mode", type=str, default="model", choices=["oracle", "model"])
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--frames", type=int, default=80)
    args = parser.parse_args()

    run_pipeline(
        scenario=args.scenario,
        mode=args.mode,
        headless=args.headless,
        demo=args.demo,
        frames=args.frames,
    )
