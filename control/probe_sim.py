"""Robotic Ultrasound Probe Kinematics & Actuator Simulator.

Owner: Maruthi (Module 3)
Component: Autonomous Probe Motion Control

Mathematical Models:
  1. Translation Kinematics (Euler integration):
       x_new = x_old + u(t) * dt
  2. Rotary Joint Model (lever arm conversion from project slides):
       delta_theta = delta_x / L
       theta_new = theta_old + delta_theta
  3. Actuator Latency Delay Buffer (1-2 frames delay)
  4. Physical Stroke Limits & Speed Saturation
"""

from collections import deque
from typing import Tuple, Dict, Any, Optional
import numpy as np


class ProbeKinematicsSimulator:
    """Simulates 1-DOF (lateral) or 2-DOF (lateral + depth) motorized ultrasound probe."""

    def __init__(
        self,
        dt: float = 0.05,                   # Sample time (seconds)
        link_length_mm: float = 150.0,       # Lever arm link length L in mm (ASSUMED)
        max_velocity_mm_s: float = 80.0,    # Max motor speed in mm/s (ASSUMED)
        range_limit_mm: float = 60.0,       # +/- travel stroke limit in mm (ASSUMED)
        actuator_delay_frames: int = 1,     # Delay in frames between command and motion (ASSUMED)
        measurement_noise_std: float = 0.5, # Gaussian sensor noise std (pixels) (ASSUMED)
        px_per_mm: float = 1.0,             # Assumed scale (1 px = 1 unit)
    ):
        self.dt = dt
        self.L = link_length_mm
        self.max_v = max_velocity_mm_s
        self.range_limit = range_limit_mm
        self.delay_frames = actuator_delay_frames
        self.noise_std = measurement_noise_std
        self.px_per_mm = px_per_mm

        # State variables: position in mm and angle in radians
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.theta_x = 0.0 # Joint angle theta
        self.theta_y = 0.0

        # Delay buffer for commanded velocities (x, y)
        self.cmd_buffer = deque([(0.0, 0.0)] * (self.delay_frames + 1), maxlen=self.delay_frames + 1)

    def step(self, cmd_vel_x: float, cmd_vel_y: float = 0.0) -> Dict[str, float]:
        """Advance probe kinematics by one time step dt.

        Args:
            cmd_vel_x: Commanded velocity in X axis (mm/s or px/s).
            cmd_vel_y: Commanded velocity in Y axis.

        Returns:
            Dict containing current probe position, angles, and noisy measurement.
        """
        # Store command in delay buffer
        self.cmd_buffer.append((cmd_vel_x, cmd_vel_y))

        # Retrieve delayed command that actually executes this cycle
        exec_vx, exec_vy = self.cmd_buffer[0]

        # Velocity clamp
        exec_vx = np.clip(exec_vx, -self.max_v, self.max_v)
        exec_vy = np.clip(exec_vy, -self.max_v, self.max_v)

        # Displacement delta_x = v * dt
        delta_x = exec_vx * self.dt
        delta_y = exec_vy * self.dt

        # Rotary joint update: delta_theta = delta_x / L; theta_new = theta_old + delta_theta
        delta_theta_x = delta_x / self.L
        delta_theta_y = delta_y / self.L
        self.theta_x += delta_theta_x
        self.theta_y += delta_theta_y

        # Position update: x_new = x_old + delta_x
        self.pos_x = np.clip(self.pos_x + delta_x, -self.range_limit, self.range_limit)
        self.pos_y = np.clip(self.pos_y + delta_y, -self.range_limit, self.range_limit)

        # Add optional measurement noise
        noise_x = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0
        noise_y = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0

        pos_px_x = self.pos_x * self.px_per_mm
        pos_px_y = self.pos_y * self.px_per_mm

        return {
            "pos_mm_x": float(self.pos_x),
            "pos_mm_y": float(self.pos_y),
            "pos_px_x": float(pos_px_x),
            "pos_px_y": float(pos_px_y),
            "pos_noisy_px_x": float(pos_px_x + noise_x),
            "pos_noisy_px_y": float(pos_px_y + noise_y),
            "theta_rad_x": float(self.theta_x),
            "theta_deg_x": float(np.degrees(self.theta_x)),
            "executed_vx": float(exec_vx),
            "executed_vy": float(exec_vy),
        }

    def reset(self):
        """Reset probe position to origin."""
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.cmd_buffer = deque([(0.0, 0.0)] * (self.delay_frames + 1), maxlen=self.delay_frames + 1)
