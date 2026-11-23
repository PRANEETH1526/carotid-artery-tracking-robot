"""Discrete PID Controller with Anti-Windup and Derivative Filtering.

Owner: Maruthi (Module 3)
Component: Autonomous Probe Motion Control

Mathematical Model:
  1. Tracking Error:
       e(t) = x_d - x(t)
  2. Proportional Term:
       P(t) = K_p * e(t)
  3. Integral Term (with clamping anti-windup):
       I(t) = I(t-1) + K_i * e(t) * dt
       I(t) = clamp(I(t), -I_max, I_max)
  4. Filtered Derivative Term (low-pass noise rejection):
       de_raw = (e(t) - e(t-1)) / dt
       de_filt(t) = alpha * de_raw + (1 - alpha) * de_filt(t-1)
       D(t) = K_d * de_filt(t)
  5. Saturated Velocity Output:
       u(t) = clamp(P(t) + I(t) + D(t), -u_max, u_max)
"""

from typing import Tuple, Dict, Any


class DiscretePID:
    """Discrete-time PID controller for 1-DOF or 2-DOF robotic probe positioning."""

    def __init__(
        self,
        kp: float = 0.65,
        ki: float = 0.12,
        kd: float = 0.08,
        dt: float = 0.05,                   # Sample interval (0.05s = 20 Hz, matching ultrasound video)
        output_limit: float = 50.0,         # Maximum commanded velocity limit (px/s or mm/s)
        integral_limit: float = 25.0,       # Anti-windup clamping threshold
        derivative_filter_alpha: float = 0.2,# Low-pass filter smoothing coefficient (0.0 to 1.0)
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)
        self.alpha = float(derivative_filter_alpha)

        # Internal controller states
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.is_first_step = True

    def compute(self, error: float) -> Tuple[float, Dict[str, float]]:
        """Compute control signal u(t) for current error e(t).

        Args:
            error: Current error signal e(t) = target - measurement.

        Returns:
            Tuple of (saturated_u, breakdown_dict)
        """
        # 1. Proportional term
        p_term = self.kp * error

        # 2. Integral term with anti-windup clamping
        self.integral += error * self.dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.ki * self.integral

        # 3. Filtered derivative term
        if self.is_first_step:
            raw_derivative = 0.0
            self.filtered_derivative = 0.0
            self.is_first_step = False
        else:
            raw_derivative = (error - self.prev_error) / self.dt
            self.filtered_derivative = (self.alpha * raw_derivative) + ((1.0 - self.alpha) * self.filtered_derivative)
        d_term = self.kd * self.filtered_derivative

        # 4. Total unsaturated command
        u_raw = p_term + i_term + d_term

        # 5. Output velocity saturation
        if u_raw > self.output_limit:
            u_saturated = self.output_limit
        elif u_raw < -self.output_limit:
            u_saturated = -self.output_limit
        else:
            u_saturated = u_raw

        self.prev_error = error

        terms = {
            "p_term": float(p_term),
            "i_term": float(i_term),
            "d_term": float(d_term),
            "u_raw": float(u_raw),
            "u_saturated": float(u_saturated),
        }
        return float(u_saturated), terms

    def reset(self):
        """Reset internal controller history."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.is_first_step = True
