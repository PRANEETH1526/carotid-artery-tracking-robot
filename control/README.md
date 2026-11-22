# Autonomous Probe Motion Control & Simulation Module

**Owner:** Maruthi  
**Component:** Closed-Loop PID Control Simulation & System Integration

---

## Purpose
Simulate the robotic ultrasound probe kinematics and closed-loop visual servoing controller. The controller consumes the tracking error from `tracking` and drives simulated 1-DOF / 2-DOF motors to keep the carotid artery centered in the ultrasound image.

---

## Interface Contract

### Input
- **Tracking Error:** $(e_x, e_y)$ in millimeters or pixels from Shabrish's `tracking.track_lumen()`.
- **Feedback Loop:** Time delta $\Delta t$ between frames.

### Output
- **Control Signal:** Motor velocities / displacement commands $(u_x, u_y)$ computed via PID controller.
- **Probe Position State:** Simulated probe position $(p_x, p_y)$ and resulting crop offset on ultrasound stream.

### Standard Function / Class Signature
```python
class PIDController:
    """Discrete PID controller for probe positioning."""
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float) -> float:
        """Compute control signal u(t) from error e(t)."""
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return u


def run_closed_loop_simulation(stream_source, unet_model, pid_params):
    """End-to-end simulation loop connecting stream -> detection -> tracking -> control."""
    pass
```

---

## End-to-End Pipeline Demo
Maruthi will integrate the entire project by pulling together:
1. `src.stream.frame_stream()` $\rightarrow$ simulated probe feed
2. `detection.detect()` $\rightarrow$ Srujan's U-Net mask
3. `tracking.track_lumen()` $\rightarrow$ Shabrish's tracking error
4. `control.PIDController.compute()` $\rightarrow$ motor velocity correction
