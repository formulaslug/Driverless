import numpy as np
from typing import Tuple


Array2D = np.ndarray


class StanleyController:
    def __init__(
        self,
        k_e: float,                    # gain for cross-track error
        k_yaw: float,                  # gain for heading error
        k_v: float,                    # speed softening term in denominator
        max_steer: float,              # steering limit (radians)
        k_throttle: float,             # gain for throttle control
        k_brake: float,                # gain for brake control
        speed_deadband: float          # m/s deadband around target speed
    ) -> None:
        self.k_e: float = k_e
        self.k_yaw: float = k_yaw
        self.k_v: float = k_v
        self.max_steer: float = max_steer
        self.k_throttle: float = k_throttle
        self.k_brake: float = k_brake
        self.speed_deadband: float = speed_deadband

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        a: float = (angle + np.pi) % (2 * np.pi) - np.pi
        return a

    def compute_control(
        self,
        x: float,
        y: float,
        yaw: float,
        v: float,
        path_xy: Array2D,  # shape (N, 2)
        v_ref: float
    ) -> Tuple[float, float, float]:
        """
        Stanley lateral + simple P speed control.

        Parameters
        ----------
        x, y : float
            Vehicle position in global frame.
        yaw : float
            Vehicle heading (rad, global frame).
        v : float
            Vehicle speed (m/s).
        path_xy : np.ndarray, shape (N, 2)
            Reference path waypoints [[x0, y0], [x1, y1], ...].
        v_ref : float
            Target speed along path (m/s).

        Returns
        -------
        delta_cmd : float
            Steering command (rad), saturated to [-max_steer, max_steer].
        throttle_cmd : float
            Throttle command [0, 1].
        brake_cmd : float
            Brake command [0, 1].
        """

        # --- 1. Find nearest point on the path ---
        dx: Array2D = path_xy[:, 0] - x
        dy: Array2D = path_xy[:, 1] - y
        d2: Array2D = dx**2 + dy**2

        nearest_idx: int = int(np.argmin(d2))
        nearest_point: Array2D = path_xy[nearest_idx]

        # --- 2. Compute path heading at nearest segment ---
        if nearest_idx < len(path_xy) - 1:
            next_point: Array2D = path_xy[nearest_idx + 1]
        else:
            # at the end, look backwards
            next_point = path_xy[nearest_idx]
            nearest_point = path_xy[nearest_idx - 1]

        path_dx: float = float(next_point[0] - nearest_point[0])
        path_dy: float = float(next_point[1] - nearest_point[1])
        path_yaw: float = float(np.arctan2(path_dy, path_dx))

        # --- 3. Heading error (vehicle vs path) ---
        heading_error: float = self._normalize_angle(path_yaw - yaw)

        # --- 4. Cross-track error ---
        # Transform vector from vehicle -> nearest point into vehicle frame
        # Vehicle frame: x forward, y to left
        map_to_vehicle_R: Array2D = np.array([
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw),  np.cos(-yaw)]
        ])

        vec_to_nearest_map: Array2D = np.array([
            nearest_point[0] - x,
            nearest_point[1] - y
        ])
        vec_to_nearest_vehicle: Array2D = map_to_vehicle_R @ vec_to_nearest_map
        cross_track_error: float = float(vec_to_nearest_vehicle[1])  # y in vehicle frame

        # --- 5. Stanley steering law (with speed softening term) ---
        eps: float = 1e-3
        denom: float = v + self.k_v + eps
        stanley_term: float = float(np.arctan2(self.k_e * cross_track_error, denom))
        delta_cmd: float = self.k_yaw * heading_error + stanley_term

        # Saturate steering command
        delta_cmd = float(np.clip(delta_cmd, -self.max_steer, self.max_steer))

        # --- 6. SIMPLE (i.e. placeholder) longitudinal P controller (throttle/brake) ---
        speed_error: float = v_ref - v

        if abs(speed_error) < self.speed_deadband:
            # within deadband: no accel, no brake
            throttle_cmd: float = 0.0
            brake_cmd: float = 0.0
        elif speed_error > 0.0:
            # need to accelerate
            throttle_cmd = float(np.clip(self.k_throttle * speed_error, 0.0, 1.0))
            brake_cmd = 0.0
        else:
            # need to decelerate
            throttle_cmd = 0.0
            brake_cmd = float(np.clip(self.k_brake * (-speed_error), 0.0, 1.0))

        return delta_cmd, throttle_cmd, brake_cmd
