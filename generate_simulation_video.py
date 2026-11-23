"""Generate High-Definition Multi-Panel Closed-Loop Simulation Video & GIF.

Creates a medical robotics simulation dashboard:
- Panel 1: B-Mode Ultrasound with U-Net Artery Segmentation & Centroid
- Panel 2: 2D Visual Servoing Convergence & Error Vector to (128, 128)
- Panel 3: Robotic Ultrasound Probe Kinematics & Joint Angle Visualization
- Panel 4: Real-Time Strip Chart (Tracking Error ex(t), PID Output u(t), and Deadband)

Multi-phase sequence:
1. Baseline Tracking (Frames 0-25)
2. Patient Motion Step Disturbance (+40 px) -> PID Servoing (Frames 26-75)
3. Pulsatile / Respiratory Sinusoid (Frames 76-115)
4. Target Dropout & Kalman Coasting Recovery (Frames 116-140)
"""

import os
import sys
import glob
import time
import math
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure project modules are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from detection.detect import detect
from tracking.tracker import KalmanLumenTracker
from control.pid import DiscretePID
from control.probe_sim import ProbeKinematicsSimulator
from control.scene_sim import MotionSceneSimulator


def draw_header(canvas: np.ndarray, frame_idx: int, total_frames: int, t_sec: float, fps: float, phase_desc: str, status: str):
    """Draw top header telemetry bar."""
    cv2.rectangle(canvas, (0, 0), (1280, 55), (15, 20, 30), -1)
    cv2.line(canvas, (0, 55), (1280, 55), (56, 214, 225), 2)

    # Title
    cv2.putText(canvas, "AUTONOMOUS CAROTID ARTERY TRACKING ROBOT | CLOSED-LOOP SIMULATION",
                (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 255), 2, cv2.LINE_AA)

    # Telemetry
    phase_text = f"PHASE: {phase_desc}"
    cv2.putText(canvas, phase_text, (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 214, 225), 1, cv2.LINE_AA)

    status_color = (52, 190, 112) if status == "TRACKING" else ((255, 174, 66) if status == "PREDICTED" else (0, 0, 255))
    cv2.putText(canvas, f"STATUS: {status}", (880, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    time_str = f"FRAME: {frame_idx:03d}/{total_frames:03d} | TIME: {t_sec:4.2f}s | {fps:4.1f} FPS"
    cv2.putText(canvas, time_str, (880, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 215), 1, cv2.LINE_AA)


def draw_ultrasound_panel(canvas: np.ndarray, frame: np.ndarray, mask: np.ndarray, cx: float, cy: float, status: str):
    """Panel 1: B-Mode Ultrasound + Segmentation Mask + Centroid."""
    px, py, pw, ph = 20, 75, 380, 380
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (22, 28, 42), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 75, 105), 1)

    # Title banner
    cv2.rectangle(canvas, (px, py), (px + pw, py + 28), (28, 36, 55), -1)
    cv2.putText(canvas, "1. B-MODE FEED & U-NET DETECTION", (px + 10, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 245, 255), 1, cv2.LINE_AA)

    # Render ultrasound image scaled to 340x340
    img_size = 340
    ox, oy = px + (pw - img_size) // 2, py + 32
    bgr = cv2.cvtColor(cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)

    scale = img_size / 256.0

    # Draw target center crosshair (128, 128)
    tx, ty = int(128.0 * scale), int(128.0 * scale)
    cv2.drawMarker(bgr, (tx, ty), (56, 214, 225), cv2.MARKER_CROSS, 18, 1)
    cv2.circle(bgr, (tx, ty), 6, (56, 214, 225), 1)

    # Draw artery segmentation contour (Green)
    if mask is not None and np.count_nonzero(mask == 255) > 0:
        mask_scaled = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, (52, 190, 112), 2)

    # Draw centroid & error vector
    cx_s, cy_s = int(cx * scale), int(cy * scale)
    marker_color = (255, 174, 66) if status == "TRACKING" else ((0, 255, 255) if status == "PREDICTED" else (0, 0, 255))
    cv2.drawMarker(bgr, (cx_s, cy_s), marker_color, cv2.MARKER_CROSS, 14, 2)
    cv2.line(bgr, (tx, ty), (cx_s, cy_s), (0, 165, 255), 1, cv2.LINE_AA)

    canvas[oy:oy + img_size, ox:ox + img_size] = bgr


def draw_visual_servoing_panel(canvas: np.ndarray, trajectory: List[Tuple[float, float]], cx: float, cy: float,
                               ex: float, ey: float, status: str, d_eq: float):
    """Panel 2: 2D Visual Servoing Convergence Space."""
    px, py, pw, ph = 420, 75, 420, 380
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (22, 28, 42), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 75, 105), 1)

    # Title banner
    cv2.rectangle(canvas, (px, py), (px + pw, py + 28), (28, 36, 55), -1)
    cv2.putText(canvas, "2. TRACKING & STATE ESTIMATION", (px + 10, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 245, 255), 1, cv2.LINE_AA)

    # Center of visual servoing polar display
    cx_disp, cy_disp = px + 210, py + 160
    radius_max = 110
    scale_factor = radius_max / 128.0

    # Concentric target zones
    cv2.circle(canvas, (cx_disp, cy_disp), radius_max, (40, 50, 70), 1)
    cv2.circle(canvas, (cx_disp, cy_disp), int(40 * scale_factor), (50, 65, 90), 1)
    cv2.circle(canvas, (cx_disp, cy_disp), int(10 * scale_factor), (35, 90, 60), 1) # Target tolerance band (10 px)
    cv2.drawMarker(canvas, (cx_disp, cy_disp), (56, 214, 225), cv2.MARKER_CROSS, 16, 1)

    # Grid axes
    cv2.line(canvas, (cx_disp - radius_max, cy_disp), (cx_disp + radius_max, cy_disp), (40, 50, 70), 1)
    cv2.line(canvas, (cx_disp, cy_disp - radius_max), (cx_disp, cy_disp + radius_max), (40, 50, 70), 1)

    # Draw past trajectory trailing path (fading color)
    n_pts = len(trajectory)
    for idx, (tx, ty) in enumerate(trajectory[-25:]):
        alpha = (idx + 1) / min(25, n_pts)
        dx = int((tx - 128.0) * scale_factor)
        dy = int((ty - 128.0) * scale_factor)
        pt_x, pt_y = cx_disp + dx, cy_disp + dy
        color = (int(56 * alpha), int(214 * alpha), int(225 * alpha))
        cv2.circle(canvas, (pt_x, pt_y), 2, color, -1)

    # Current position dot
    dx_now = int((cx - 128.0) * scale_factor)
    dy_now = int((cy - 128.0) * scale_factor)
    curr_pt = (cx_disp + dx_now, cy_disp + dy_now)
    marker_color = (255, 174, 66) if status == "TRACKING" else ((0, 255, 255) if status == "PREDICTED" else (0, 0, 255))
    cv2.circle(canvas, curr_pt, 6, marker_color, -1)
    cv2.circle(canvas, curr_pt, 8, (255, 255, 255), 1)
    cv2.line(canvas, (cx_disp, cy_disp), curr_pt, (0, 165, 255), 2, cv2.LINE_AA)

    # Numeric Metrics Box
    my = py + 285
    cv2.rectangle(canvas, (px + 15, my), (px + pw - 15, my + 80), (16, 22, 34), -1)
    cv2.rectangle(canvas, (px + 15, my), (px + pw - 15, my + 80), (45, 58, 82), 1)

    cv2.putText(canvas, f"Centroid (cx, cy): ({cx:5.1f}, {cy:5.1f}) px", (px + 25, my + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 230, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Error Vector e:   ({ex:+5.1f}, {ey:+5.1f}) px", (px + 25, my + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (56, 214, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Equiv. Diameter:  {d_eq:5.1f} px  | Filter: Kalman", (px + 25, my + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 185, 205), 1, cv2.LINE_AA)


def draw_probe_kinematics_panel(canvas: np.ndarray, probe_x: float, theta_deg: float, u_cmd: float, pid_parts: Dict[str, float]):
    """Panel 3: Robotic Ultrasound Probe & Joint Kinematics."""
    px, py, pw, ph = 860, 75, 400, 380
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (22, 28, 42), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 75, 105), 1)

    # Title banner
    cv2.rectangle(canvas, (px, py), (px + pw, py + 28), (28, 36, 55), -1)
    cv2.putText(canvas, "3. PROBE KINEMATICS & MOTOR ACTUATION", (px + 10, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 245, 255), 1, cv2.LINE_AA)

    # Draw neck surface and probe schematic
    cx_probe = px + 200 + int(probe_x * 1.5) # scaled probe motion
    cx_probe = max(px + 70, min(px + pw - 70, cx_probe))

    skin_y = py + 200
    # Skin line (curved tissue surface)
    cv2.line(canvas, (px + 20, skin_y), (px + pw - 20, skin_y), (140, 120, 100), 2)
    cv2.putText(canvas, "PATIENT NECK SURFACE", (px + 25, skin_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 120, 100), 1)

    # Ultrasound beam cone (semi-transparent green beam)
    beam_len = 55
    angle_rad = math.radians(theta_deg)
    cone_w = 26
    bx0, by0 = cx_probe, skin_y
    bx1 = int(bx0 - cone_w * math.cos(angle_rad) + beam_len * math.sin(angle_rad))
    by1 = int(by0 + beam_len * math.cos(angle_rad) + cone_w * math.sin(angle_rad))
    bx2 = int(bx0 + cone_w * math.cos(angle_rad) + beam_len * math.sin(angle_rad))
    by2 = int(by0 + beam_len * math.cos(angle_rad) - cone_w * math.sin(angle_rad))

    pts = np.array([[bx0, by0], [bx1, by1], [bx2, by2]], np.int32)
    cv2.fillPoly(canvas, [pts], (20, 50, 35))
    cv2.polylines(canvas, [pts], True, (52, 190, 112), 1)

    # Probe body (blue-grey rounded rectangle)
    probe_top = skin_y - 80
    cv2.rectangle(canvas, (cx_probe - 20, probe_top), (cx_probe + 20, skin_y), (65, 85, 120), -1)
    cv2.rectangle(canvas, (cx_probe - 20, probe_top), (cx_probe + 20, skin_y), (120, 150, 200), 2)
    cv2.drawMarker(canvas, (cx_probe, probe_top), (56, 214, 225), cv2.MARKER_TRIANGLE_UP, 10, 2)
    cv2.putText(canvas, "US PROBE", (cx_probe - 26, probe_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 230), 1)

    # Carotid artery target circle underneath skin
    art_x = px + 200
    art_y = skin_y + 40
    cv2.circle(canvas, (art_x, art_y), 15, (50, 50, 180), -1)
    cv2.circle(canvas, (art_x, art_y), 15, (80, 80, 240), 2)
    cv2.putText(canvas, "CAROTID ARTERY", (art_x - 45, art_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 220), 1)

    # Telemetry data readouts
    my = py + 285
    cv2.rectangle(canvas, (px + 15, my), (px + pw - 15, my + 80), (16, 22, 34), -1)
    cv2.rectangle(canvas, (px + 15, my), (px + pw - 15, my + 80), (45, 58, 82), 1)

    cv2.putText(canvas, f"Motor u(t):     {u_cmd:+5.1f} px/s (Kp: {pid_parts.get('p', 0.0):+4.1f}, Ki: {pid_parts.get('i', 0.0):+4.1f})",
                (px + 22, my + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (56, 214, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Probe Pos X:    {probe_x:+5.1f} px (1-DOF linear carriage)",
                (px + 22, my + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 230, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Joint Angle th: {theta_deg:+5.2f} deg (Rotary beam align)",
                (px + 22, my + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 185, 205), 1, cv2.LINE_AA)


def draw_strip_chart(canvas: np.ndarray, error_history: List[float], control_history: List[float], max_len: int = 140):
    """Panel 4: Real-Time Strip Chart (Oscilloscope) for Error & PID Output."""
    px, py, pw, ph = 20, 475, 1240, 230
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 24, 36), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 75, 105), 1)

    # Header bar
    cv2.rectangle(canvas, (px, py), (px + pw, py + 24), (26, 34, 50), -1)
    cv2.putText(canvas, "4. REAL-TIME CLOSED-LOOP PERFORMANCE (ERROR CONVERGENCE & PID MOTOR DAMPING)",
                (px + 15, py + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 245, 255), 1, cv2.LINE_AA)

    # Chart bounds
    gx, gy, gw, gh = px + 60, py + 35, pw - 80, ph - 55
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (12, 16, 24), -1)

    # Zero error line (center line)
    zero_y = gy + gh // 2
    cv2.line(canvas, (gx, zero_y), (gx + gw, zero_y), (80, 100, 130), 1)
    cv2.putText(canvas, "0 px (Target Center)", (gx + 5, zero_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 140, 170), 1)

    # Deadband tolerance zone (+/- 5 px)
    y_scale = gh / 100.0 # scale +/- 50 px
    db_y1 = int(zero_y - 5.0 * y_scale)
    db_y2 = int(zero_y + 5.0 * y_scale)
    cv2.rectangle(canvas, (gx, db_y1), (gx + gw, db_y2), (25, 45, 35), -1)
    cv2.putText(canvas, "Deadband (+/-5 px)", (gx + gw - 130, zero_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (52, 190, 112), 1)

    # Grid reference lines (+30, -30 px)
    y_p30 = int(zero_y - 30.0 * y_scale)
    y_m30 = int(zero_y + 30.0 * y_scale)
    cv2.line(canvas, (gx, y_p30), (gx + gw, y_p30), (25, 32, 45), 1)
    cv2.line(canvas, (gx, y_m30), (gx + gw, y_m30), (25, 32, 45), 1)
    cv2.putText(canvas, "+30 px", (gx - 45, y_p30 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 115, 140), 1)
    cv2.putText(canvas, "-30 px", (gx - 45, y_m30 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 115, 140), 1)

    # Plot error curve (Orange/Red) and control curve (Cyan)
    if len(error_history) > 1:
        x_step = gw / float(max_len)
        err_pts = []
        u_pts = []
        for i in range(len(error_history)):
            curr_x = int(gx + i * x_step)
            # Clamped error to +/- 50
            e_val = max(-48.0, min(48.0, error_history[i]))
            e_y = int(zero_y - e_val * y_scale)
            err_pts.append((curr_x, e_y))

            # Scaled control signal
            u_val = max(-48.0, min(48.0, control_history[i]))
            u_y = int(zero_y - u_val * (y_scale * 0.7))
            u_pts.append((curr_x, u_y))

        for j in range(len(err_pts) - 1):
            cv2.line(canvas, err_pts[j], err_pts[j + 1], (0, 165, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, u_pts[j], u_pts[j + 1], (56, 214, 225), 1, cv2.LINE_AA)

    # Legend
    cv2.rectangle(canvas, (gx + 15, gy + 10), (gx + 260, gy + 32), (18, 22, 32), -1)
    cv2.line(canvas, (gx + 25, gy + 21), (gx + 45, gy + 21), (0, 165, 255), 2)
    cv2.putText(canvas, "Tracking Error ex(t)", (gx + 52, gy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 235, 245), 1)
    cv2.line(canvas, (gx + 160, gy + 21), (gx + 180, gy + 21), (56, 214, 225), 1)
    cv2.putText(canvas, "PID u(t)", (gx + 187, gy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 235, 245), 1)


def generate_simulation():
    print("=" * 65)
    print("GENERATING COMPREHENSIVE CAROTID TRACKING SIMULATION VIDEO & GIF")
    print("=" * 65)

    test_imgs = sorted(glob.glob("dataset/processed/test/images/*.png"))
    test_masks = sorted(glob.glob("dataset/processed/test/masks/*.png"))
    if not test_imgs:
        test_imgs = sorted(glob.glob("carotid_project/dataset/processed/test/images/*.png"))
        test_masks = sorted(glob.glob("carotid_project/dataset/processed/test/masks/*.png"))

    assert len(test_imgs) > 0, "No test images found!"

    base_img = cv2.imread(test_imgs[10], cv2.IMREAD_GRAYSCALE)
    base_mask = cv2.imread(test_masks[10], cv2.IMREAD_GRAYSCALE)

    dt = 0.05 # 20 Hz
    scene = MotionSceneSimulator(base_img, base_mask, dt=dt)
    tracker = KalmanLumenTracker(dt=dt, max_coast_frames=6)
    pid = DiscretePID(kp=0.90, ki=0.05, kd=0.04, dt=dt, output_limit=50.0, integral_limit=25.0)
    probe = ProbeKinematicsSimulator(dt=dt, max_velocity_mm_s=80.0, actuator_delay_frames=1)

    total_frames = 140
    dashboard_frames = []
    error_history = []
    control_history = []
    trajectory = []

    curr_probe_x = 0.0

    output_dir = "results" if os.path.exists("results") else "carotid_project/results"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "simulation_dashboard.mp4")
    gif_path = os.path.join(output_dir, "simulation_dashboard.gif")
    pipeline_gif_path = os.path.join(output_dir, "pipeline_demo.gif")

    print(f"Total simulation frames: {total_frames} (at 20 FPS = 7.0 seconds)")
    print(f"Output Video: {video_path}")
    print(f"Output GIF:   {gif_path}")

    # Set up OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))

    t_start = time.time()

    for k in range(total_frames):
        t_sec = k * dt

        # Define multi-phase scenario
        if k < 25:
            scenario = "step"
            phase_desc = "Baseline Resting Alignment (Steady State)"
            ext_offset = 0.0
            dropout = False
        elif k < 75:
            scenario = "step"
            phase_desc = "Patient Step Motion (+40 px displacement) -> PID Recovery"
            ext_offset = 40.0
            dropout = False
        elif k < 115:
            scenario = "sinusoidal"
            phase_desc = "Respiratory / Pulsatile Sinusoid (A=15 px, f=0.5 Hz)"
            ext_offset = 15.0 * math.sin(2 * math.pi * 0.5 * (k - 75) * dt)
            dropout = False
        else:
            scenario = "step"
            phase_desc = "Acoustic Dropout (5 frames) -> Kalman Coasting (PREDICTED)"
            ext_offset = 0.0
            dropout = (118 <= k <= 124)

        # 1. Update scene motion
        net_offset = ext_offset - curr_probe_x
        # Shift frame
        h, w = base_img.shape
        M = np.float32([[1, 0, net_offset], [0, 1, 0]])
        frame = cv2.warpAffine(base_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        gt_mask = cv2.warpAffine(base_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 2. Artery detection (simulate dropout during frames 118-124)
        if dropout:
            mask = np.zeros_like(frame) # Complete loss of echo
        else:
            mask = detect(frame)

        # 3. Kalman Tracking
        track_state = tracker.update(mask)
        cx = track_state["cx"]
        cy = track_state["cy"]
        status = track_state["status"]
        d_eq = track_state["diameter_px"]
        trajectory.append((cx, cy))

        # 4. PID error & control
        ex = float(cx - 128.0)
        ey = float(cy - 128.0)
        u, breakdown = pid.compute(ex)

        error_history.append(ex)
        control_history.append(u)

        # 5. Probe kinematics
        probe_state = probe.step(cmd_vel_x=u)
        curr_probe_x = probe_state["pos_px_x"]
        theta_deg = probe_state["theta_deg_x"]

        # 6. Assemble Full Dashboard Canvas (1280x720)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:] = (12, 16, 24) # Dark slate background

        draw_header(canvas, k + 1, total_frames, t_sec, 20.0, phase_desc, status)
        draw_ultrasound_panel(canvas, frame, mask, cx, cy, status)
        draw_visual_servoing_panel(canvas, trajectory, cx, cy, ex, ey, status, d_eq)
        draw_probe_kinematics_panel(canvas, curr_probe_x, theta_deg, u, breakdown)
        draw_strip_chart(canvas, error_history, control_history, max_len=total_frames)

        # Write to MP4 video
        vw.write(canvas)
        dashboard_frames.append(canvas)

        if (k + 1) % 20 == 0 or (k + 1) == total_frames:
            print(f"  Frame {k+1:03d}/{total_frames:03d} processed | Error: {ex:+5.1f} px | PID u: {u:+5.1f} px/s")

    vw.release()
    t_elapsed = time.time() - t_start
    print(f"\nMP4 Video generation complete in {t_elapsed:.1f}s: {video_path}")

    # Generate optimized GIF (scaled down to 640x360 for high performance & presentation embedding)
    print("Generating animated GIF (640x360 at 15 FPS)...")
    pil_frames = []
    # Take every second frame for compact GIF (70 frames total)
    for f in dashboard_frames[::2]:
        f_small = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
        f_rgb = cv2.cvtColor(f_small, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(f_rgb))

    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=66, # ~15 FPS
        loop=0,
        optimize=True,
    )
    print(f"Animated GIF saved: {gif_path}")

    # Also update pipeline_demo.gif
    pil_frames[0].save(
        pipeline_gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=66,
        loop=0,
        optimize=True,
    )
    print(f"Updated pipeline demo GIF: {pipeline_gif_path}")

    return video_path, gif_path


if __name__ == "__main__":
    generate_simulation()
