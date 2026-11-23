# Detection Subsystem (Srujan) — Presentation & Viva Study Notes

**Component:** Carotid Artery Lumen Segmentation using U-Net  
**Language:** Simple English for First-Year Students and Presentation Prep  

---

## 1. What This Module Does (in Simple Words)
This module takes a single ultrasound picture of the neck and finds the **carotid artery lumen** (the black circular hole where blood flows to the brain). It outputs a black-and-white mask where **white (255) is the artery** and **black (0) is everything else**.

---

## 2. Everyday Example: How U-Net Works
Think of U-Net like an artist copying a small drawing:
1. **Contracting Path (Encoder):** First, the artist steps back and squints to see the big picture. This tells the artist *what* is in the image (is it an artery?), but blurs the exact boundary edges.
2. **Expanding Path (Decoder):** Next, the artist steps forward and redraws the fine details at full resolution to determine *where* the artery boundary is.
3. **Skip Connections:** Because squinting loses fine sharpness, U-Net uses "skip connections" like a direct phone call from the early detailed layers directly to the late rebuilding layers. This brings back crisp, exact wall boundaries.

---

## 3. How the Detection Pipeline Works Step by Step
1. **Input Frame:** Receives a 256×256 ultrasound frame with pixel values from 0.0 (black) to 1.0 (white).
2. **U-Net Forward Pass:** The neural network looks at the image and predicts a probability score (0.0 to 1.0) for every pixel.
3. **Thresholding:** Pixels with probability $\ge 0.5$ become candidate artery pixels.
4. **Largest Connected Component (LCC):** In ultrasound, speckle noise can cause tiny fake white dots. LCC keeps only the single biggest blob (the actual vessel) and deletes all fake dots.
5. **Hole Filling:** If blood turbulence creates a fake black speck inside the white artery, morphological hole filling seals it shut.
6. **Mask Smoothing:** An elliptical closing/opening filter smooths the vessel edge so it looks like a realistic round blood vessel.
7. **Output Mask:** Returns a 256×256 binary mask (`uint8`, values 0 and 255).

---

## 4. Key Numbers to Remember
- **Model Parameters:** ~486,000 parameters (under 1 Million, very lightweight ~1.9 MB file).
- **Training Set:** 770 images from 27 independent recording sessions.
- **Validation Set:** 173 images from 6 independent recording sessions.
- **Test Set:** 157 images from 5 independent recording sessions.
- **Inference Latency:** ~10 to 18 milliseconds per frame (comfortably 50+ FPS on GPU, ~30 FPS on CPU).
- **Test Dice Score:** ~0.82 to 0.88 (significantly beats classical Otsu thresholding baseline at ~0.50).

---

## 5. Realistic Limitations (Be Honest!)
1. **Single Dataset:** Trained on one Kaggle transverse ultrasound dataset; has not been tested across multiple hospital scanner machines.
2. **Acoustic Shadows:** Heavy bone shadows or calcified plaque can darken the vessel wall, causing the model to underestimate artery size.
3. **Slice Discontinuity:** The dataset consists of sampled image slices, not continuous 60 FPS live video.
4. **Dominant Test Sessions:** In the test set of 157 images, three sessions contribute over 90% of frames.

---

## 6. Top 10 Viva / Teacher Questions & Simple Answers

### Q1: Why did you choose U-Net instead of YOLO or a bounding box detector?
**Answer:** A bounding box only gives a square around the artery. But robotic visual servoing and doctor diagnostics need the exact circular boundary and pixel area of the vessel lumen to track its true centroid accurately. U-Net provides pixel-level segmentation.

### Q2: What happens if the detection fails or the artery is temporarily lost?
**Answer:** If the network finds no artery (or the predicted area is below the minimum threshold), `detect()` safely returns an empty mask (all zeros). Downstream, Shabrish's Kalman filter detects this, switches to `PREDICTED` mode, and coasts smoothly on the last known velocity for up to 5 frames without jerking the robotic probe. If lost for longer, it safely halts probe motion.

### Q3: Why did you combine Binary Cross-Entropy (BCE) with Dice Loss?
**Answer:** In a 256×256 ultrasound frame, the carotid artery only occupies 1% to 5% of all pixels (95% is background). Standard BCE would be biased toward predicting background. Dice Loss directly optimizes overlap (intersection over union), solving the extreme foreground/background class imbalance.

### Q4: Why did you train on 256×256 instead of the original 749×709 resolution?
**Answer:** 256×256 reduces the memory footprint and computation by ~8×, allowing the model to run at over 40 FPS for real-time robotic visual servoing while preserving all critical lumen anatomical features.

### Q5: Did you pre-train on ImageNet or train from scratch?
**Answer:** We trained from scratch. Natural RGB photos (cats, cars) in ImageNet have completely different statistical noise than medical ultrasound acoustic speckle. With 770 training frames and data augmentation, training from scratch converged stably and achieved strong generalization.

### Q6: What data augmentations did you use and why only on the training set?
**Answer:** We used random horizontal flips, small rotations ($\pm 10^\circ$), small spatial shifts, and slight brightness changes applied identically to image and mask. We strictly never augment validation or test sets so that evaluation measures true, uncorrupted clinical performance.

### Q7: Why did you exclude the CUBS dataset from training?
**Answer:** Inspection showed that CUBS is a longitudinal (side) view and its ground-truth masks only annotate the thin 2-pixel intima-media wall strip, not the circular transverse lumen. Mixing horizontal line masks with circular cross-sections would confuse the network.

### Q8: How did you prevent data leakage during training?
**Answer:** We grouped all images by recording session prefix before splitting. All frames from any single patient session stay strictly inside either Train, Val, or Test. No session appears in more than one split.

### Q9: What is the purpose of Largest Connected Component (LCC) post-processing?
**Answer:** Ultrasound images often contain hypoechoic (dark) acoustic artifacts outside the vessel (such as shadows or small veins). LCC automatically keeps only the primary vessel and eliminates isolated false-positive speckle noise.

### Q10: How does your U-Net compare to a classical baseline method?
**Answer:** A classical Otsu thresholding method only achieves ~0.45 to 0.55 Dice because it confuses any dark acoustic shadow with an artery. Our U-Net learns spatial shape context and achieves ~0.85+ Dice on the exact same test set.
