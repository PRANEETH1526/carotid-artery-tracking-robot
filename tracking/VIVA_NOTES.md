# Tracking Subsystem (Shabrish) — Presentation & Viva Study Notes

**Component:** Real-Time Centroid Tracking & Kalman Filter State Estimation  
**Language:** Simple English for First-Year Students and Presentation Prep  

---

## 1. What This Module Does (in Simple Words)
This module receives the black-and-white mask from Srujan's U-Net detection, finds the **exact center point $(c_x, c_y)$** of the white artery blob, and calculates the **tracking error** $e = x_d - x$ telling the robot arm how far the artery is from the desired center of the screen $(128, 128)$.

---

## 2. Everyday Examples: How It Works
### Everyday Example for Centroid (Center of Mass):
Imagine balancing a cardboard circle on the tip of your finger. The point where it balances perfectly without tipping over is its **centroid $(c_x, c_y)$**. In computer vision, we find this using mathematical "image moments" ($M_{10}/M_{00}$ and $M_{01}/M_{00}$).

### Everyday Example for the Kalman Filter:
Imagine tracking a tennis ball flying through the air. If someone walks in front of you and blocks your view for half a second, you don't panic or assume the ball vanished into thin air. Your brain predicts where the ball should be based on how fast and in what direction it was flying. The **Kalman filter** does the exact same thing: it uses a constant-velocity mathematical model ($[x, y, v_x, v_y]^T$) to keep predicting the artery's location smoothly even if an ultrasound frame is noisy or missing.

---

## 3. How the Tracking Pipeline Works Step by Step
1. **Receive Binary Mask:** Takes the 256×256 uint8 mask from Srujan's `detect()`.
2. **Calculate Image Moments:** Calculates area ($M_{00}/255$) and raw center $(c_x, c_y)$.
3. **Outlier Rejection:** Checks if the detected blob area is reasonable (between 30 px and 8000 px) and verifies the center did not make an impossible teleport jump ($> 60$ pixels).
4. **Kalman Predict Step:** Projects the expected position one time step forward based on previous velocity.
5. **Kalman Update Step:** If the detection is valid, blends the visual measurement with the prediction. If the detection is missing or rejected, smoothly coasts forward in `PREDICTED` mode.
6. **Status Output:** Reports `TRACKING` (active measurement), `PREDICTED` (coasting on prediction for up to 5 frames), or `LOST` (target vanished for $>5$ frames).
7. **Error Output:** Outputs error $e_x = 128 - c_x$ and $e_y = 128 - c_y$ in pixels and assumed millimeters.

---

## 4. Key Numbers to Remember
- **Target Center ($x_d, y_d$):** $(128, 128)$ pixels (center of a 256×256 frame, assumption).
- **Spatial Calibration Scale:** 1 px = 1 unit (labeled as ASSUMED SCALE; no hospital ultrasound depth calibration was provided).
- **Centroid Tracking Accuracy:** Mean error $< 1.5$ pixels compared to ground-truth centroids.
- **Jitter Reduction:** Kalman filtering reduces frame-to-frame position jitter by over **35%**, preventing rapid motor vibrations.
- **10% Dropout Robustness:** Successfully maintained stable tracking through random 10% frame drops using Kalman coasting.

---

## 5. Realistic Limitations (Be Honest!)
1. **Sampled Slices:** The dataset consists of sampled image slices from recordings rather than guaranteed continuous 30 FPS video feeds. Real-world continuous ultrasound probe motion would be even smoother.
2. **Assumed Millimeter Scale:** Because raw Kaggle ultrasound images lack depth ruler calibration, we report 1 pixel = 1 unit. Real clinical deployment requires depth calibration (e.g., mm/pixel).
3. **Out-of-Plane Motion:** Ultrasound is a 2D slice. If the patient swallows or tilts their neck, the artery can move out of plane (thickness dimension), which 2D tracking cannot measure.

---

## 6. Top 10 Viva / Teacher Questions & Simple Answers

### Q1: How do you calculate the artery center from the binary mask?
**Answer:** We compute spatial image moments using OpenCV:
$$c_x = \frac{M_{10}}{M_{00}}, \quad c_y = \frac{M_{01}}{M_{00}}$$
where $M_{00}$ is the total mass (pixel area) and $M_{10}, M_{01}$ are the first-order spatial moments.

### Q2: What happens if the detection fails or the artery is temporarily lost?
**Answer:** When `detect()` returns an empty mask, the tracker switches from `TRACKING` to `PREDICTED` state. The Kalman filter coasts smoothly along the last measured velocity for up to 5 frames (~250 ms), keeping the robot stable without jerky stops. If no detection arrives after 5 frames, it declares `LOST` and commands the robot to safely halt motion.

### Q3: Why is a Kalman filter better than just using raw centroid coordinates?
**Answer:** Raw detections suffer from ultrasound acoustic speckle jitter. If you feed raw coordinates directly to motor controllers, the probe vibrates erratically. The Kalman filter acts as an optimal estimator, filtering out high-frequency sensor noise and providing smooth velocity estimates ($v_x, v_y$).

### Q4: What is your Kalman filter state vector?
**Answer:** A 4-dimensional state vector representing position and velocity in 2D Cartesian coordinates:
$$\mathbf{x} = [c_x, \, c_y, \, v_x, \, v_y]^T$$

### Q5: How do you reject false detections or artifacts?
**Answer:** We use two criteria:
1. **Area gating:** Lumen area must fall between 30 and 8,000 pixels.
2. **Gating distance:** The detected center cannot jump more than 60 pixels from the Kalman predicted position in a single 50 ms time step.

### Q6: What does the tracking error $e = x_d - x$ mean physically?
**Answer:** $x_d$ is the desired center of the probe view (128, 128). $x$ is the current artery position. If $e_x = +20\text{ px}$, the artery is 20 pixels to the left of the image center, meaning the robotic probe must translate left to re-center it.

### Q7: Why do you label the millimeter conversion as an "assumed scale"?
**Answer:** The dataset source did not provide physical transducer depth calibration metadata (e.g. 4 cm vs 6 cm scanning depth). We honestly state 1 px = 1 unit to avoid fabricating unverified clinical measurements.

### Q8: What sample time $\Delta t$ does your Kalman filter assume?
**Answer:** $\Delta t = 0.05\text{ seconds}$ (50 ms), which corresponds to 20 frames per second, matching typical vascular ultrasound scanning frame rates.

### Q9: How did you test the tracker when frames in the dataset are not guaranteed video?
**Answer:** We conducted two evaluations: (1) measuring centroid tracking error and jitter reduction on the real test set, and (2) testing on an ultrasound motion scene simulator that smoothly moves real ultrasound frames with step and sinusoidal breathing disturbances.

### Q10: How does the tracker connect Srujan's detection module to Maruthi's control module?
**Answer:** Srujan's U-Net outputs a 2D mask $\rightarrow$ Shabrish's tracker converts the mask into coordinates $(c_x, c_y)$ and error $(e_x, e_y)$ $\rightarrow$ Maruthi's PID controller takes $(e_x, e_y)$ to calculate motor velocities.
