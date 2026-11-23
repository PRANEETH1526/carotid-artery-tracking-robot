# Autonomous Carotid Artery Tracking Robot: Dataset & Preprocessing Subsystem

This repository contains the software-only dataset preparation, image preprocessing pipeline, dataset splitting module, and live ultrasound frame stream simulator for the **Autonomous Carotid Artery Tracking Robot** mini-project.

The outputs generated here serve as the upstream data pipeline for downstream teammates:
- **Srujan (Detection / Segmentation):** Receives the standardized `dataset/processed/{train,val,test}` directories and the standalone `preprocess(img)` function to train and evaluate the U-Net segmentation network.
- **Shabrish & Maruthi (Tracking & PID Control Simulation):** Receive the real-time `frame_stream(path, loop, fps)` generator to simulate live B-mode ultrasound feeds driving the probe visual servoing and PID control loop.

---

## 1. Project Directory Structure

```text
carotid_project/
├── dataset/
│   ├── raw/
│   │   ├── images/                       # 1,100 raw transverse B-mode ultrasound PNGs (749x709)
│   │   └── masks/                        # 1,100 ground truth lumen binary masks (749x709)
│   └── processed/
│       ├── all/                          # 1,100 preprocessed images & masks (256x256, 0-255 PNG)
│       │   ├── images/                   # img_0001.png ... img_1100.png
│       │   ├── masks/                    # mask_0001.png ... mask_1100.png
│       │   └── metadata.csv              # Full dataset index, session IDs, and size mapping
│       ├── train/                        # 787 images & masks (27 sessions, 71.5%)
│       │   ├── images/, masks/
│       │   └── train_metadata.csv
│       ├── val/                          # 218 images & masks (6 sessions, 19.8%)
│       │   ├── images/, masks/
│       │   └── val_metadata.csv
│       └── test/                         # 95 images & masks (5 sessions, 8.6%)
│           ├── images/, masks/
│           └── test_metadata.csv
├── src/
│   ├── __init__.py                       # Python package marker
│   ├── preprocess.py                     # Single-image & batch preprocessing pipeline
│   ├── split_dataset.py                  # Session-wise train/val/test splitter (+ augmentation)
│   ├── stream.py                         # Live ultrasound frame generator for tracking simulation
│   └── visualize.py                      # Multi-stage comparisons, overlays & split charts
├── reports/
│   ├── inspection/                       # Phase 1 exploratory overlays & sample analysis
│   ├── preprocessing_comparison.png      # 5 samples comparing Raw -> Median -> CLAHE
│   ├── sample_overlays.png               # 5 samples showing preprocessed image + lumen overlay
│   ├── dataset_split_counts.png          # Presentation-ready bar chart of split distributions
│   └── stream_preview.png                # Live ultrasound stream HUD preview snapshot
├── requirements.txt                      # Minimum Python dependencies
└── README.md                             # Project documentation and interface contract
```

---

## 2. Dataset Information & Analysis

### Primary Dataset (Used for Training & Evaluation)
- **Source:** Kaggle "Common Carotid Artery Ultrasound Images"
- **Modality:** B-mode Transverse Ultrasound (Cross-Section View)
- **Anatomy:** Common Carotid Artery (CCA) and adjacent internal jugular vein
- **Annotated Structure:** **Artery Lumen** (the blood-filled echo-free interior lumen represented as a solid binary blob)
- **Image Count:** 1,100 image-mask pairs
- **Raw Dimensions:** 749 × 709 pixels
- **License & Data Note:** Dataset source = Kaggle "Common Carotid Artery Ultrasound Images", license = TO BE CONFIRMED by the repo owner (verify against original Kaggle competition / author terms).

### Why CUBS (`archive (2)`) Was Excluded From Training
- **Dataset C (CUBS)** contains 2,176 images and masks in a **longitudinal view** (lateral orientation).
- Inspection revealed that CUBS ground-truth masks annotate only the thin **intima-media thickness (IMT) complex along the far vessel wall** (a thin horizontal line/strip), **NOT** the circular cross-sectional artery lumen.
- Mixing longitudinal wall strips with transverse circular lumen masks would confuse the U-Net boundary detector. Therefore, CUBS was left untouched in its raw folder and excluded from the training split.

### Dataset B (`archive (1)`) — Unannotated Carotid Ultrasound
- Contains 6,890 unannotated B-mode ultrasound images in transverse cross-section view.
- Because it lacks ground truth masks, it cannot be used for supervised segmentation training.
- However, because it contains genuine transverse carotid ultrasound acquisitions, it is integrated into `stream.py` (`--use-dataset-b`) as an extended live video feed source for tracking and PID control testing.

---

## 3. Burned-In Text and UI Marker Analysis

Visual and pixel-level inspection across the 1,100 raw images confirmed:
1. **No Patient Information or Hospital Watermarks:** The raw images have already been pre-cropped to the acoustic sector.
2. **No Calipers or Depth Rulers:** The sector borders are clean (bottom boundary values are 0-1).
3. **Channel Uniformity:** Although saved with 3 color channels (BGR), pixel values are nearly identical across channels (small deviations <= 23 are due to compression artifacts). They are converted strictly to single-channel grayscale via `cv2.COLOR_BGR2GRAY`.

---

## 4. Preprocessing Pipeline Specification

To standardize image inputs for U-Net training and real-time tracking, the pipeline applies:

| Step | Operation | Parameters | Rationale |
|---|---|---|---|
| 1 | **Grayscale Conversion** | `cv2.COLOR_BGR2GRAY` | Discards redundant 3-channel depth |
| 2 | **Spatial Resizing** | `256 × 256` (`cv2.INTER_LINEAR`) | Standardized square tensor for deep learning backbones |
| 3 | **Speckle Denoising** | `cv2.medianBlur(img, 5)` | Eliminates high-frequency ultrasound acoustic speckle while preserving sharp vessel borders |
| 4 | **Contrast Enhancement** | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` | Boosts dynamic range of the dark lumen against hyper-reflective muscular walls |
| 5 | **Tensor Normalization** | `astype(float32) / 255.0` | Normalizes pixel values into standard range `[0.0, 1.0]` |

### Mask Preprocessing
- **Resizing:** `cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)` to prevent non-binary interpolated boundary edge bleed.
- **Binarization:** `cv2.threshold(..., 127, 255, cv2.THRESH_BINARY)` ensuring ground truth is strictly `{0, 255}` uint8 (0 = background, 255 = artery lumen).

---

## 5. Dataset Splitting & Subject Leakage Prevention

### Session-Wise (Patient-Wise) Split
Ultrasound image sequences contain multiple sequential video frames (slices) extracted from the same recording session. A naive random split would place adjacent slices from the same subject into both training and test sets (causing massive artificial data leakage).

To guarantee strict generalizability:
- We grouped all 1,100 images into their **38 independent recording sessions** (parsed via filename prefix, e.g. `202201121748100022VAS_...`).
- We split by session using a **fixed random seed (`seed=1`)**:

| Split | Sessions | Images | Percentage | Leakage | Unmatched |
|---|---|---|---|---|---|
| **Train** | 27 sessions | 770 images | 70.0% | None (0%) | 0 |
| **Validation** | 6 sessions | 173 images | 15.7% | None (0%) | 0 |
| **Test** | 5 sessions | 157 images | 14.3% | None (0%) | 0 |
| **Total** | **38 sessions** | **1,100 images** | **100.0%** | **0%** | **0** |

> **Note on Seed Selection (`seed=1`):** With 38 unique ultrasound recording sessions (each contributing 1 to 90 consecutive frames), partitions must split whole sessions. Seed `1` was chosen specifically to ensure a well-balanced test set (14.3%, 157 images) and validation set (15.7%, 173 images) closely matching the 70/15/15 target with zero session overlap and zero data leakage.

---

## 6. How to Run Each Script

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Batch Preprocessing
Processes all raw images and masks into `dataset/processed/all/` with standardized naming (`img_0001.png` <-> `mask_0001.png`) and creates `metadata.csv`:
```bash
python src/preprocess.py
```

### 2. Dataset Splitting
Partitions preprocessed data into `train`, `val`, and `test` sets:
```bash
# Standard split (770 train, 173 val, 157 test)
python src/split_dataset.py --seed 1

# Optional: Generate augmented training set (+10/-10 deg rotation, horizontal flips, brightness jitter)
python src/split_dataset.py --seed 1 --augment
```

### 3. Live Ultrasound Stream Simulator
Simulates a real-time ultrasound probe video feed:
```bash
# Stream from the processed test set (20 FPS throttle, loops continuously)
python src/stream.py --demo

# Stream from unannotated Dataset B (100 evenly-spaced demo frames inside dataset/dataset_b_demo/)
python src/stream.py --demo --use-dataset-b

# Stream with custom frame rate and frame limit
python src/stream.py --demo --fps 30 --max-frames 150
```
*Keyboard controls during demo:* Press `Space` to pause/resume, `q` or `ESC` to exit.

### 4. Presentation Visualizations
Generates high-resolution figures in `reports/` for your project report and slide deck:
```bash
python src/visualize.py
```

---

## 7. Interface Contract for Teammates

### Srujan (Segmentation & U-Net Training)
1. **Training Data Location:**
   - Training images: `dataset/processed/train/images/`
   - Training masks: `dataset/processed/train/masks/`
   - Validation images: `dataset/processed/val/images/`
   - Validation masks: `dataset/processed/val/masks/`
   - Test images: `dataset/processed/test/images/`
   - Test masks: `dataset/processed/test/masks/`
2. **File Naming & Alignment:**
   - Every `img_XXXX.png` matches exactly with `mask_XXXX.png`.
   - Dimensions are guaranteed to be `(256, 256)` single-channel grayscale uint8.
   - Masks are strictly binary: `0` (background) and `255` (carotid lumen).
3. **Inference Preprocessing Function:**
   ```python
   from src.preprocess import preprocess

   # During evaluation or deployment on any raw B-mode frame:
   input_tensor = preprocess(raw_frame, size=(256, 256))  # returns float32 in [0.0, 1.0]
   ```

### Shabrish & Maruthi (Tracking & PID Control Simulation)
1. **Live Feed Generator:**
   Use `frame_stream` from `src.stream` to simulate continuous incoming ultrasound frames:
   ```python
   from src.stream import frame_stream

   # Initialize video feed from test set or raw sequence
   feed = frame_stream(
       path="dataset/processed/test/images",
       loop=True,        # Keep looping continuously for long simulations
       fps=25.0,         # Match 25 FPS probe acquisition speed
       as_uint8=False    # Yields normalized float32 [0.0, 1.0] ready for U-Net
   )

   for frame in feed:
       # 1. Run U-Net inference on frame
       pred_mask = unet_model.predict(frame)
       
       # 2. Extract lumen centroid (cx, cy)
       # 3. Compute tracking error: error_x = 128 - cx, error_y = 128 - cy
       # 4. Update PID controller to drive simulated robot probe motors
   ```


---

## 8. Team and Workflow

### Team Roles & Responsibility Matrix

| Teammate | Subsystem / Role | Primary Directory | Key Deliverables |
|---|---|---|---|
| **Praneeth** | Data Engineering, Preprocessing & Presentation | `dataset/`, `src/`, `reports/` | Ultrasound dataset curation, CLAHE/denoising pipeline, session-wise split, live stream simulator, presentation figures |
| **Srujan** | Lumen Detection & Segmentation | `detection/` | U-Net model training, loss tuning (Dice + BCE), `detect(frame) -> mask` inference function |
| **Shabrish** | Vessel Tracking & State Estimation | `tracking/` | Lumen centroid tracking, error computation ($e_x = x_d - x$), pixel-to-mm conversion, lost-target detection |
| **Maruthi** | Control & Simulation Integration | `control/` | Discrete PID controller, probe kinematics simulation, end-to-end closed-loop visual servoing demo |

---

### Getting Started (Clone & Run)

```bash
# 1. Clone the repository
git clone https://github.com/PRANEETH1526/carotid-artery-tracking-robot.git
cd carotid-artery-tracking-robot

# 2. Create and activate a Python virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the live ultrasound stream demo to verify setup
python src/stream.py --demo
```

---

### Dataset Note in Git
- **Excluded from Git:** The raw image/mask archive (`dataset/raw/`) and unpartitioned intermediate data (`dataset/processed/all/`) are excluded via `.gitignore` to keep the repository lightweight.
- **Included in Git:** The complete, preprocessed **train**, **validation**, and **test** datasets (`dataset/processed/{train,val,test}/`) are committed directly to Git, along with the 100-frame extended stream demo slice (`dataset/dataset_b_demo/`). Teammates can start model training immediately without downloading raw archives.

---

### Collaboration Rules & Git Workflow
1. **Branching Strategy:**
   - Each teammate must work on their dedicated feature branch:
     - Srujan: `detection-srujan`
     - Shabrish: `tracking-shabrish`
     - Maruthi: `control-maruthi`
   - Open a **Pull Request (PR)** into `main` for review before merging.
   - **Nobody pushes directly to `main`**, except Praneeth for shared dataset/docs updates.
2. **Fixed Interface Contract:**
   - **Do NOT change the data format** ($256\times256$ grayscale float32 `[0.0, 1.0]`, masks strictly binary `{0, 255}`).
   - **Do NOT alter the session-wise split** or random seed (`seed=1`) without notifying all teammates, as downstream test metrics must remain comparable.
### Model Weights Note
- General model weights are excluded by default, but the lightweight trained checkpoint `detection/models/unet_best.pt` (5.9 MB) is committed directly to allow immediate out-of-the-box execution of the closed-loop tracking pipeline without retraining.

---

## 9. Subsystem Execution & Closed-Loop Demo

All four subsystems have been fully implemented, validated, and integrated into a unified software-in-the-loop autonomous carotid artery tracking robot pipeline.

### Quick Start: Running the Full Closed-Loop Pipeline

```bash
# Run headless closed-loop pipeline over test session (generates HUD frames and summary metrics)
python run_pipeline.py --headless

# Run interactive visual servoing demo with live HUD overlay
python run_pipeline.py --demo --session 2026-02-09-17-08-30

# Run complete pytest verification suite (all 4 subsystems + interface contract)
pytest tests/test_all.py -v
```

### Module Verification Commands

| Module | Responsible | Execution Command | Key Outputs |
|---|---|---|---|
| **Detection** | Srujan | `python detection/evaluate.py` | `reports/detection_training_curves.png`<br>`reports/detection_test_samples.png`<br>`reports/detection_failure_analysis.md` |
| **Tracking** | Shabrish | `python tracking/evaluate_tracking.py` | `reports/tracking_trajectories.png` |
| **Control** | Maruthi | `python control/evaluate_control.py` | `reports/control_response.png` |
| **Integration** | Praneeth | `python run_pipeline.py --headless` | `results/pipeline_demo.gif`<br>`results/results.json`<br>`results/RESULTS.md` |

### Key Benchmark Summary

| Subsystem | Metric | Measured Value | Target / Requirement |
|---|---|---|---|
| **Detection (U-Net)** | Test Dice Score | **0.9486 ± 0.0255** | $\ge 0.85$ (Otsu baseline: 0.0484) |
| | Test IoU | **0.9033 ± 0.0426** | $\ge 0.75$ |
| | Inference Latency | **50.83 ms** (~19.7 FPS) | $< 100\text{ ms}$ (CPU) |
| **Tracking (Kalman)** | Mean Centroid Error | **20.68 px** | Sub-centimeter tracking |
| | Jitter Reduction | **80.6%** ($15.74 \to 3.05\text{ px}$) | Eliminates motor chatter |
| | 10% Dropout Recovery | **100% Passed** | Coasting in `PREDICTED` state |
| **Control (PID)** | Step Settling Time | **3.45 s** | $< 5.0\text{ s}$ |
| | Steady-State Error | **1.71 px** | $< 5.0\text{ px}$ |
| | Servoing Stability | Stable across Step, Sine, Drift | No divergence |
| **Closed-Loop** | End-to-End Latency | **57.91 ms** (~17.3 FPS) | Real-time ($\ge 15\text{ FPS}$) |
| | Unit Tests | **6 / 6 Passed** | 100% interface contract adherence |

