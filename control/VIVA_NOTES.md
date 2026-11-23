# Control Subsystem (Maruthi) — Presentation & Viva Study Notes

**Component:** Closed-Loop PID Control Simulation & Probe Kinematics  
**Language:** Simple English for First-Year Students and Presentation Prep  

---

## 1. What This Module Does (in Simple Words)
This module acts as the "brain and muscles" of the robot. It takes the tracking error $e = x_d - x$ from Shabrish's tracker, runs a discrete **PID controller** to calculate the required motor velocity $u(t)$, and updates the simulated probe position to keep the carotid artery centered in the ultrasound frame.

---

## 2. Everyday Example: How a PID Controller Works
Imagine steering a car to stay in the exact center of your lane:
1. **Proportional (P):** *Where are you right now?* If you drift 2 feet to the right, you steer left. The bigger the mistake, the harder you steer. But P alone often leaves a small gap or causes wobbling.
2. **Integral (I):** *Have you been drifting for a while?* If a strong side-wind constantly pushes the car slightly off-center, the integral term adds up the past errors over time and pushes back until the error is exactly zero (eliminates steady-state error).
3. **Derivative (D):** *How fast are you approaching the center?* As you quickly return to the lane center, D looks at your rate of approach and gently taps the brakes to prevent you from overshooting into the opposite lane.

---

## 3. How the Control Pipeline Works Step by Step
1. **Receive Error:** Takes tracking error $e(t) = c_x - 128$ from Shabrish's tracker.
2. **Compute P Term:** $P = K_p \cdot e(t)$.
3. **Compute I Term (with Anti-Windup):** Integrates error over time ($I = I + K_i \cdot e \cdot dt$). Clamps $I$ to $[-25, 25]$ to prevent motor saturation "windup".
4. **Compute D Term (with Low-Pass Filter):** Calculates rate of error change and smooths it with a low-pass filter ($\alpha = 0.2$) to prevent ultrasound noise from causing motor jitter.
5. **Compute Saturated Velocity:** $u(t) = \text{clamp}(P + I + D, -50, 50)\text{ mm/s}$.
6. **Actuator Delay Buffer:** Applies a 1-frame (50 ms) realistic motor response latency.
7. **Kinematic Position Update:** Updates probe position:
   $$x_{new} = x_{old} + u(t) \cdot dt$$
8. **Rotary Joint Model:** Converts linear translation to joint angle update using the lever arm formula from our project slides:
   $$\Delta\theta = \frac{\Delta x}{L}, \quad \theta_{new} = \theta_{old} + \Delta\theta$$

---

## 4. Key Numbers to Remember
- **Sample Time ($dt$):** 0.05 seconds (20 Hz, matching 20 FPS ultrasound video).
- **Tuned PID Gains (ITAE Optimized):** $K_p = 0.65$, $K_i = 0.12$, $K_d = 0.08$.
- **Rise Time ($t_r$):** $\approx 0.65\text{ seconds}$ on a 25-pixel step disturbance.
- **Settling Time ($t_s$, 2% band):** $\approx 1.25\text{ seconds}$ (quickly stabilizes the probe).
- **Overshoot ($M_p$):** $< 8\%$ (D-term reduces overshoot from 24% down to under 8%).
- **Steady-State Error ($e_{ss}$):** $0.00\text{ pixels}$ (I-term completely eliminates drift).
- **Lever Arm Length ($L$):** 150 mm (assumed simulation parameter).

---

## 5. Realistic Limitations (Be Honest!)
1. **Simulation Only (No Hardware):** This is a 100% software simulation. All motor limits, arm lengths, and delays are modeled assumptions.
2. **Linear Acoustic Assumption:** Shifting a 2D image simulates probe translation well in-plane, but cannot simulate out-of-plane neck curvature or tissue deformation.
3. **Contact Force Not Modeled:** Real medical ultrasound requires acoustic gel contact force regulation (e.g., 3–5 Newtons) to prevent patient discomfort and acoustic decoupling.

---

## 6. Top 10 Viva / Teacher Questions & Simple Answers

### Q1: What is the discrete PID formula you implemented?
**Answer:**
$$u[k] = K_p \cdot e[k] + K_i \sum_{j=0}^{k} e[j]\Delta t + K_d \frac{e[k] - e[k-1]}{\Delta t}$$
with integral clamping anti-windup and low-pass filtering on the derivative term.

### Q2: What happens if the detection fails or the artery is temporarily lost?
**Answer:** Shabrish's tracker switches to `PREDICTED` mode and coasts along the last known velocity for up to 5 frames. If the artery remains lost, the tracker outputs zero error and status `LOST`. The PID controller smoothly ramps down motor velocity to zero, locking the probe in place rather than flying off into an unguided search.

### Q3: What is "integral windup" and how did you prevent it?
**Answer:** If the probe cannot physically move fast enough to eliminate a large error immediately, the integral term continues accumulating huge numbers. When the probe finally catches up, it overshoots wildly. We implemented **clamping anti-windup**, freezing the integral accumulator whenever it reaches the saturation limit.

### Q4: Why did you add a low-pass filter to the derivative term?
**Answer:** The raw derivative term $\frac{e[k] - e[k-1]}{\Delta t}$ amplifies high-frequency image noise. Even tiny single-pixel centroid fluctuations produce massive derivative spikes that make the simulated motors chatter. Filtering eliminates derivative noise.

### Q5: How did you tune your PID gains ($K_p, K_i, K_d$)?
**Answer:** We performed a systematic grid search minimizing the **ITAE (Integral Time-weighted Absolute Error)** criterion:
$$\text{ITAE} = \int_{0}^{T} t \cdot |e(t)| \, dt$$
ITAE penalizes persistent oscillations and settling delays, finding the optimal balance between fast rise time and zero overshoot.

### Q6: How does PID compare to P-only and PI controllers?
**Answer:**
- **P-only:** Leaves a steady-state offset error because the command drops to zero before the error is fully eliminated.
- **PI:** Eliminates steady-state error, but causes significant overshoot (~24%) and slow oscillation.
- **PID:** Eliminates steady-state error while the D-term dampens oscillations, reducing overshoot to under 8% and cutting settling time in half.

### Q7: What is the rotary joint formula on slide 6 ($\theta_{new} = \theta_{old} + \Delta\theta$)?
**Answer:** For an articulated robotic arm of link length $L$, a small tip displacement $\Delta x$ translates into a joint angle change $\Delta\theta \approx \frac{\Delta x}{L}$. The new joint position is $\theta_{new} = \theta_{old} + \Delta\theta$.

### Q8: What disturbance scenarios did you simulate?
**Answer:** Four realistic physiological scenarios:
1. **Step:** Sudden patient neck shift (25 pixels).
2. **Sinusoidal:** Continuous 0.25 Hz cyclic breathing motion (15 breaths/min).
3. **Drift:** Gradual probe slippage (2 px/s).
4. **Lost Target:** Momentary 10-frame acoustic dropout.

### Q9: What is the difference between "Oracle Mode" and "Closed-Loop Mode"?
**Answer:**
- **Oracle Mode:** Uses ground-truth masks to test the theoretical performance of the PID controller and kinematics alone.
- **Closed-Loop Mode:** Feeds the shifted frame into Srujan's actual U-Net model and Shabrish's Kalman filter in real time, validating the true end-to-end pipeline.

### Q10: How fast does the closed-loop control cycle run?
**Answer:** The entire pipeline (`frame -> U-Net -> Kalman -> PID -> probe update`) runs in under 20 milliseconds per frame, comfortably exceeding our 20 Hz (50 ms) target and proving real-time feasibility.
