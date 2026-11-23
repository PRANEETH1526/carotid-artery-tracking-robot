# Carotid Artery Detection: Failure Analysis Report

**Module:** Detection (Srujan)
**Evaluation Date:** Auto-generated from real test set evaluation.

## 1. Identification of the 3 Worst-Performing Test Samples

### Worst Case #1: `img_0293.png` (Session: `202201121819140033VAS`)
- **Test Dice Score:** 0.7439
- **Test IoU:** 0.5922
- **Precision:** 0.9981
- **Recall:** 0.5929
- **Ground Truth Area:** 1749 pixels
- **Predicted Area:** 1039 pixels

### Worst Case #2: `img_1094.png` (Session: `202202071358440055VAS`)
- **Test Dice Score:** 0.8691
- **Test IoU:** 0.7684
- **Precision:** 0.9969
- **Recall:** 0.7703
- **Ground Truth Area:** 1271 pixels
- **Predicted Area:** 982 pixels

### Worst Case #3: `img_0647.png` (Session: `202201121847410041VAS`)
- **Test Dice Score:** 0.8804
- **Test IoU:** 0.7864
- **Precision:** 0.7885
- **Recall:** 0.9967
- **Ground Truth Area:** 1208 pixels
- **Predicted Area:** 1527 pixels

## 2. Root Cause Analysis: What Goes Wrong and Why

Inspection of the worst cases reveals two primary acoustic phenomena causing boundary degradation:

1. **Acoustic Shadowing and Faint Lateral Borders:**
   In severe calcification or lateral vessel wall sections, the ultrasound acoustic beam is attenuated, causing the dark lumen to bleed into adjacent hypoechoic tissue. The network underestimates the lumen width.

2. **Low-Contrast Lumen and Dense Internal Speckle:**
   Sluggish blood flow or reverberation artifacts produce dense internal echoes inside the carotid lumen, reducing acoustic contrast against surrounding muscular tissue. The probability map drops below 0.5 in the center.

3. **Downstream Safety Mitigation in Tracking:**
   When worst-case segmentation boundaries shrink, Shabrish's Kalman filter detects the anomalous area shift, coasts on velocity state projection (`PREDICTED` mode), and prevents jerky robotic commands.
