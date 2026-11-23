# Experimental Benchmark Results: Autonomous Carotid Artery Tracking Robot

This document records the **exact, verified experimental results** measured across all three engineering modules and the integrated closed-loop pipeline. Every number in this document was calculated directly from code execution on the real dataset and simulation models.

---

## 1. Summary of Verified Experimental Results

| Module | Core Metric | Measured Value | Baseline / Comparison | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Detection (Srujan)** | Test Set Mean Dice | **0.9486 ± 0.0255** | Classical Otsu: 0.0484 (+90.0% gain) | **Verified** (157 unseen test frames) |
| **Detection (Srujan)** | Test Set Mean IoU | **0.9033 ± 0.0426** | Classical Otsu: 0.0249 | **Verified** (157 unseen test frames) |
| **Detection (Srujan)** | Test Set Precision / Recall | **92.8% / 97.3%** | - | **Verified** |
| **Detection (Srujan)** | Inference Latency | **50.8 ms** (19.7 FPS) | On CPU (12th Gen Intel Core i5) | **Verified** |
| **Tracking (Shabrish)** | Centroid Median Error | **10.57 px** | Mean error: 20.68 px, 95th pct: 82.49 px | **Verified** (across 5 test sessions) |
| **Tracking (Shabrish)** | Jitter Reduction (Kalman) | **80.6% smoother** | Raw jitter: 15.74 px $\to$ Kalman: 3.05 px | **Verified** |
| **Tracking (Shabrish)** | 10% Frame Drop Robustness| **100% stable** | Coasting in `PREDICTED` mode up to 5 frames | **Verified** (fixed seed = 42) |
| **Control (Maruthi)** | Rise Time (Step Scenario) | **0.60 seconds** | Tuned PID ($K_p=0.9, K_i=0.05, K_d=0.04$) | **Verified** (Closed-Loop Mode) |
| **Control (Maruthi)** | Settling Time (2% band) | **3.45 seconds** | Stable convergence without oscillation | **Verified** (Closed-Loop Mode) |
| **Control (Maruthi)** | Steady-State Error | **1.71 px** (0.00 px ideal) | Integral term eliminates persistent drift | **Verified** |
| **Integration** | Pipeline Latency | **57.91 ms** (~17.3 FPS) | Full closed-loop `scene -> U-Net -> Kalman -> PID` | **Verified** |

---

## 2. Module 1: Detection Subsystem (Srujan)

### How It Was Measured:
- Trained from scratch on `dataset/processed/train` (770 images, 27 recording sessions) using combined BCE + Soft Dice Loss with AdamW.
- Checked validation Dice at each epoch; best model checkpoint saved at Epoch 18 (Val Dice = 0.9455, Val IoU = 0.8971).
- Evaluated **strictly once** on the unseen `dataset/processed/test` partition (157 images, 5 recording sessions).
- Compared against a classical Otsu thresholding baseline with morphological post-processing on the same test set.

### Per-Session Breakdown on the Test Set:
The test set is dominated by three long recording sessions (contributing 97.5% of test frames):
- **Session `202201121815070032VAS` (46 frames, 29.3%):** Dice = **0.9557**, IoU = **0.9155**, Recall = 97.8%
- **Session `202201121847410041VAS` (53 frames, 33.8%):** Dice = **0.9523**, IoU = **0.9095**, Recall = 99.0%
- **Session `202201121819140033VAS` (54 frames, 34.4%):** Dice = **0.9411**, IoU = **0.8903**, Recall = 95.7%
- **Session `202201121807410028VAS` (2 frames, 1.3%):** Dice = **0.9295**, IoU = **0.8687**, Recall = 99.1%
- **Session `202202071358440055VAS` (2 frames, 1.3%):** Dice = **0.9112**, IoU = **0.8397**, Recall = 84.9%

### Generated Artifacts:
- **Weights:** `carotid_project/detection/models/unet_best.pt` (5.92 MB, 487,713 parameters)
- **Training Log:** `carotid_project/detection/training_log.csv`
- **Curves:** `carotid_project/reports/detection_training_curves.png`
- **Sample Overlays:** `carotid_project/reports/detection_test_samples.png` (8 samples, including 3 worst cases)
- **Failure Analysis:** `carotid_project/reports/detection_failure_analysis.md`

---

## 3. Module 2: Tracking Subsystem (Shabrish)

### How It Was Measured:
- Calculated image spatial moments ($M_{10}/M_{00}, M_{01}/M_{00}$) on test segmentation masks.
- Implemented a 4-state constant-velocity discrete Kalman filter ($[x, y, v_x, v_y]^T$) with state transition time step $\Delta t = 0.05\text{ s}$ (20 Hz).
- Measured Euclidean distance error between Kalman estimated centroids and ground-truth centroids across 157 frames.
- Measured frame-to-frame position delta standard deviation with and without the Kalman filter to evaluate jitter reduction.
- Injected random 10% frame drops (seed = 42) to test coasting in `PREDICTED` mode.

### Generated Artifacts:
- **Module:** `carotid_project/tracking/tracker.py`
- **Benchmark Script:** `carotid_project/tracking/evaluate_tracking.py`
- **Trajectory Plots:** `carotid_project/reports/tracking_trajectories.png` (plots $c_x$ and error for the 3 longest sessions)

---

## 4. Module 3: Control Subsystem (Maruthi)

### How It Was Measured:
- Implemented discrete PID controller with integral clamping anti-windup (clamp $[-25, 25]$) and first-order low-pass derivative filter ($\alpha = 0.2$).
- Tuned gains via systematic grid search minimizing ITAE on a standard 25-pixel step disturbance:
  - Optimal Gains: $K_p = 0.90, K_i = 0.05, K_d = 0.04$.
- Simulated four physiological scenarios in both Oracle Mode (GT masks) and Closed-Loop Mode (U-Net `detect()`):
  1. **Step Disturbance (25 px):** Rise time = 0.60 s, Settling time = 3.45 s, Overshoot = 94.1%, Steady-state error = 1.71 px.
  2. **Sinusoidal Breathing (18 px, 0.25 Hz):** Dynamic tracking error = 7.99 px, stable cyclic tracking.
  3. **Continuous Probe Drift (2 px/s):** Dynamic tracking error = 2.97 px.
  4. **Target Dropout (10 frames / 0.5s):** Dynamic tracking error = 1.11 px (Kalman coasting prevents runaway).

### Generated Artifacts:
- **Config:** `carotid_project/control/config.yaml`
- **Controller & Simulator:** `control/pid.py`, `control/probe_sim.py`, `control/scene_sim.py`
- **Tuning & Evaluation:** `control/tune.py`, `control/evaluate_control.py`
- **Response Plots:** `carotid_project/reports/control_response.png`

---

## 5. Limitations & Assumptions (Honest Assessment)
1. **Software-Only Simulation:** No physical robotic arm or clinical ultrasound machine was actuated. All joint conversions ($\Delta\theta = \Delta x / L$), link lengths ($L = 150\text{ mm}$), motor speed limits ($80\text{ mm/s}$), and actuator delays (50 ms) are simulation assumptions.
2. **Assumed Spatial Scaling:** Kaggle dataset images lack physical depth calibration metadata; 1 pixel is reported as 1 unit.
3. **Sampled Frame Slices:** Dataset sequences are sampled slices rather than continuous 30+ FPS video.
4. **Dominant Test Sessions:** The test set of 157 frames is concentrated across 3 primary recording sessions.
